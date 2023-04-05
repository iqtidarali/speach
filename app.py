from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
from audio_foundation_models import *
import gradio as gr

AUDIO_CHATGPT_PREFIX = """Audio ChatGPT
AUdio ChatGPT can not directly read audios, but it has a list of tools to finish different audio synthesis tasks. Each audio will have a file name formed as "audio/xxx.wav". When talking about audios, Audio ChatGPT is very strict to the file name and will never fabricate nonexistent files. 
AUdio ChatGPT is able to use tools in a sequence, and is loyal to the tool observation outputs rather than faking the audio content and audio file name. It will remember to provide the file name from the last tool observation, if a new audio is generated.
Human may provide Audio ChatGPT with a description. Audio ChatGPT should generate audios according to this description rather than directly imagine from memory or yourself."
TOOLS:
------
Audio ChatGPT  has access to the following tools:"""

AUDIO_CHATGPT_FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:
```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```
When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""

AUDIO_CHATGPT_SUFFIX = """You are very strict to the filename correctness and will never fake a file name if not exists.
You will remember to provide the audio file name loyally if it's provided in the last tool observation.
Begin!
Previous conversation history:
{chat_history}
New input: {input}
Thought: Do I need to use a tool? {agent_scratchpad}"""

def cut_dialogue_history(history_memory, keep_last_n_words = 500):
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f"history_memory:{history_memory}, n_tokens: {n_tokens}")
    if n_tokens < keep_last_n_words:
        return history_memory
    else:
        paragraphs = history_memory.split('\n')
        last_n_tokens = n_tokens
        while last_n_tokens >= keep_last_n_words:
            last_n_tokens = last_n_tokens - len(paragraphs[0].split(' '))
            paragraphs = paragraphs[1:]
        return '\n' + '\n'.join(paragraphs)

class ConversationBot:
    def __init__(self, load_dict):
        print("Initializing AudioGPT")
        self.tools = []
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')
        self.models = dict()
        for class_name, device in load_dict.items():
            self.models[class_name] = globals()[class_name](device=device)
        for class_name, instance in self.models.items():
            for e in dir(instance):
                if e.startswith('inference'):
                    func = getattr(instance, e)
                    self.tools.append(Tool(name=func.name, description=func.description, func=func))

    def run_text(self, text, state):
        print("===============Running run_text =============")
        print("Inputs:", text, state)
        print("======>Previous memory:\n %s" % self.agent.memory)
        self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=500)
        res = self.agent({"input": text})
        if res['intermediate_steps'] == []:
            print("======>Current memory:\n %s" % self.agent.memory)
            response = res['output']
            state = state + [(text, response)]
            print("Outputs:", state)
            return state, state, gr.Audio.update(visible=False), gr.Image.update(visible=False), gr.Button.update(visible=False)
        else:
            tool = res['intermediate_steps'][0][0].tool
            if tool == "Generate Image From User Input Text":
                res['output'] = res['output'].replace("\\", "/")
                response = re.sub('(image/\S*png)', lambda m: f'![](/file={m.group(0)})*{m.group(0)}*', res['output'])
                state = state + [(text, response)]
                print(f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
                      f"Current Memory: {self.agent.memory.buffer}")
                return state, state, gr.Audio.update(visible=False), gr.Image.update(visible=False), gr.Button.update(visible=False)
            elif tool == "Generate Text From The Audio" or tool == "Transcribe speech":
                print("======>Current memory:\n %s" % self.agent.memory)
                response = re.sub('(image/\S*png)', lambda m: f'![](/file={m.group(0)})*{m.group(0)}*', res['output'])
                image_filename = res['intermediate_steps'][0][1]
                #response = res['output'] + f"![](/file={image_filename})*{image_filename}*"
                state = state + [(text, response)]
                print("Outputs:", state)
                return state, state, gr.Audio.update(visible=False), gr.Image.update(visible=False), gr.Button.update(visible=False)
            elif tool == "Audio Inpainting":
                audio_filename = res['intermediate_steps'][0][0].tool_input
                image_filename = res['intermediate_steps'][0][1]
               # self.is_visible(True)
                print("======>Current memory:\n %s" % self.agent.memory)
                print(res)
                response = res['output']
                state = state + [(text, response)]
                print("Outputs:", state)
                return state, state, gr.Audio.update(value=audio_filename,visible=True), gr.Image.update(value=image_filename,visible=True), gr.Button.update(visible=True)
            print("======>Current memory:\n %s" % self.agent.memory)
            response = re.sub('(image/\S*png)', lambda m: f'![](/file={m.group(0)})*{m.group(0)}*', res['output'])
            audio_filename = res['intermediate_steps'][0][1]
            state = state + [(text, response)]
            print("Outputs:", state)
            return state, state, gr.Audio.update(value=audio_filename,visible=True), gr.Image.update(visible=False), gr.Button.update(visible=False)

    def run_image_or_audio(self, file, state, txt):
        file_type = file.name[-3:]
        if file_type == "wav":
            print("===============Running run_audio =============")
            print("Inputs:", file, state)
            print("======>Previous memory:\n %s" % self.agent.memory)
            audio_filename = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
            audio_load = whisper.load_audio(file.name)
            soundfile.write(audio_filename, audio_load, samplerate = 16000)
            description = self.models['A2T'].inference(audio_filename)
            Human_prompt = "\nHuman: provide an audio named {}. The description is: {}. This information helps you to understand this audio, but you should use tools to finish following tasks, " \
                           "rather than directly imagine from my description. If you understand, say \"Received\". \n".format(audio_filename, description)
            AI_prompt = "Received.  "
            self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
            # AI_prompt = "Received.  "
            # self.agent.memory.buffer = self.agent.memory.buffer + 'AI: ' + AI_prompt
            print("======>Current memory:\n %s" % self.agent.memory)
            #state = state + [(f"<audio src=audio_filename controls=controls></audio>*{audio_filename}*", AI_prompt)]
            state = state + [(f"*{audio_filename}*", AI_prompt)]
            print("Outputs:", state)
            return state, state, txt + ' ' + audio_filename + ' ', gr.Audio.update(value=audio_filename,visible=True)
        else:
            print("===============Running run_image =============")
            print("Inputs:", file, state)
            print("======>Previous memory:\n %s" % self.agent.memory)
            image_filename = os.path.join('image', str(uuid.uuid4())[0:8] + ".png")
            print("======>Auto Resize Image...")
            img = Image.open(file.name)
            width, height = img.size
            ratio = min(512 / width, 512 / height)
            width_new, height_new = (round(width * ratio), round(height * ratio))
            width_new = int(np.round(width_new / 64.0)) * 64
            height_new = int(np.round(height_new / 64.0)) * 64
            img = img.resize((width_new, height_new))
            img = img.convert('RGB')
            img.save(image_filename, "PNG")
            print(f"Resize image form {width}x{height} to {width_new}x{height_new}")
            description = self.models['ImageCaptioning'].inference(image_filename)
            Human_prompt = "\nHuman: provide an audio named {}. The description is: {}. This information helps you to understand this audio, but you should use tools to finish following tasks, " \
                           "rather than directly imagine from my description. If you understand, say \"Received\". \n".format(image_filename, description)
            AI_prompt = "Received.  "
            self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
            print("======>Current memory:\n %s" % self.agent.memory)
            state = state + [(f"![](/file={image_filename})*{image_filename}*", AI_prompt)]
            print(f"\nProcessed run_image, Input image: {image_filename}\nCurrent state: {state}\n"
                  f"Current Memory: {self.agent.memory.buffer}")
            return state, state, txt + ' ' + image_filename + ' ', gr.Audio.update(visible=False)

    def inpainting(self, state, audio_filename, image_filename):
        print("===============Running inpainting =============")
        print("Inputs:", state)
        print("======>Previous memory:\n %s" % self.agent.memory)
        # inpaint = Inpaint(device="cpu")
        new_image_filename, new_audio_filename = self.models['Inpaint'].predict(audio_filename, image_filename)       
        AI_prompt = "Here are the predict audio and the mel spectrum." + f"*{new_audio_filename}*" + f"![](/file={new_image_filename})*{new_image_filename}*"
        self.agent.memory.buffer = self.agent.memory.buffer + 'AI: ' + AI_prompt
        print("======>Current memory:\n %s" % self.agent.memory)
        state = state + [(f"Audio Inpainting", AI_prompt)]
        print("Outputs:", state)
        return state, state, gr.Image.update(visible=False), gr.Audio.update(value=new_audio_filename, visible=True), gr.Button.update(visible=False)
    def clear_audio(self):
        return gr.Audio.update(value=None, visible=False)
    def clear_image(self):
        return gr.Image.update(value=None, visible=False)
    def clear_button(self):
        return gr.Button.update(visible=False)
    def init_agent(self, openai_api_key):
        self.llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={'prefix': AUDIO_CHATGPT_PREFIX, 'format_instructions': AUDIO_CHATGPT_FORMAT_INSTRUCTIONS, 'suffix': AUDIO_CHATGPT_SUFFIX}, )
        return gr.update(visible = True)



if __name__ == '__main__': 
    bot = ConversationBot({'TTS_OOD': 'cuda:0',
                           'ImageCaptioning': 'cuda:0',
                           'T2A': 'cuda:0',
                           'I2A': 'cuda:0',
                           'TTS': 'cpu',
                           'T2S': 'cpu',
                           'ASR': 'cuda:0',
                           'A2T': 'cpu',
                           'Inpaint': 'cuda:0'
                           })
    with gr.Blocks(css="#chatbot {overflow:auto; height:500px;}") as demo:
        with gr.Row():
            openai_api_key_textbox = gr.Textbox(
                placeholder="Paste your OpenAI API key here to start AudioGPT(sk-...) and press Enter ↵️",
                show_label=False,
                lines=1,
                type="password",
            )
        with gr.Row():
            gr.Markdown("## AudioGPT")
        chatbot = gr.Chatbot(elem_id="chatbot", label="AudioGPT")
        state = gr.State([])
        with gr.Row(visible = False) as input_raws:
            with gr.Column(scale=0.7):
                txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter, or upload an image").style(container=False)
            with gr.Column(scale=0.1, min_width=0):
                run = gr.Button("🏃‍♂️Run")
            with gr.Column(scale=0.1, min_width=0):
                clear = gr.Button("🔄Clear️")
            with gr.Column(scale=0.1, min_width=0):
                btn = gr.UploadButton("🖼️Upload", file_types=["image","audio"])
        with gr.Row():        
            with gr.Column():
                outaudio = gr.Audio(visible=False)
        with gr.Row():           
            with gr.Column():
                show_mel = gr.Image(type="filepath",tool='sketch',visible=False)
        with gr.Row():           
            with gr.Column():        
                run_button = gr.Button("Predict Masked Place",visible=False)
        gr.Examples(
            examples=["Generate an audio of a dog barking",
                      "Generate an audio of this image",
                      "Can you describe the audio with text?",
                      "Generate a speech with text 'here we go'",
                      "Generate an image of a cat",
                      "I want to inpaint this audio",
                     # "generate a piece of singing voice. Text sequence is 小酒窝长睫毛AP是你最美的记号. Note sequence is C#4/Db4 | F#4/Gb4 | G#4/Ab4 | A#4/Bb4 F#4/Gb4 | F#4/Gb4 C#4/Db4 | C#4/Db4 | rest | C#4/Db4 | A#4/Bb4 | G#4/Ab4 | A#4/Bb4 | G#4/Ab4 | F4 | C#4/Db4. Note duration sequence is 0.407140 | 0.376190 | 0.242180 | 0.509550 0.183420 | 0.315400 0.235020 | 0.361660 | 0.223070 | 0.377270 | 0.340550 | 0.299620 | 0.344510 | 0.283770 | 0.323390 | 0.360340."
                      ],
            inputs=txt
        )

        openai_api_key_textbox.submit(bot.init_agent, [openai_api_key_textbox], [input_raws])    
        txt.submit(bot.run_text, [txt, state], [chatbot, state, outaudio, show_mel, run_button])
        txt.submit(lambda: "", None, txt)
        run.click(bot.run_text, [txt, state], [chatbot, state, outaudio, show_mel, run_button])
        run.click(lambda: "", None, txt)
        btn.upload(bot.run_image_or_audio, [btn, state, txt], [chatbot, state, txt, outaudio])
        run_button.click(bot.inpainting, [state, outaudio, show_mel], [chatbot, state, show_mel, outaudio, run_button])
        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)
        clear.click(lambda:None, None, txt)
        clear.click(bot.clear_button, None, run_button)
        clear.click(bot.clear_image, None, show_mel)
        clear.click(bot.clear_audio, None, outaudio)
        demo.launch(server_name="0.0.0.0", server_port=7860)