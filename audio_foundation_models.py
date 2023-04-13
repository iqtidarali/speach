import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'NeuralSeq'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'text_to_audio/Make_An_Audio'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'audio_detection'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mono2binaural'))
import matplotlib
import librosa
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPSegProcessor, CLIPSegForImageSegmentation
import torch
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import re
import uuid
import soundfile
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
import cv2
import einops
from einops import repeat
from pytorch_lightning import seed_everything
import random
from ldm.util import instantiate_from_config
from ldm.data.extract_mel_spectrogram import TRANSFORMS_16000
from pathlib import Path
from vocoder.hifigan.modules import VocoderHifigan
from vocoder.bigvgan.models import VocoderBigVGAN
from ldm.models.diffusion.ddim import DDIMSampler
from wav_evaluation.models.CLAPWrapper import CLAPWrapper
from inference.svs.ds_e2e import DiffSingerE2EInfer
from audio_to_text.inference_waveform import AudioCapModel
import whisper
from text_to_speech.TTS_binding import TTSInference
from inference.svs.ds_e2e import DiffSingerE2EInfer
from inference.tts.GenerSpeech import GenerSpeechInfer
from utils.hparams import set_hparams
from utils.hparams import hparams as hp
from utils.os_utils import move_file
import scipy.io.wavfile as wavfile
from audio_infer.utils import config as detection_config
from audio_infer.pytorch.models import PVT
from src.models import BinauralNetwork
from sound_extraction.model.LASSNet import LASSNet
from sound_extraction.utils.stft import STFT
from sound_extraction.utils.wav_io import load_wav, save_wav
from target_sound_detection.src import models as tsd_models
from target_sound_detection.src.models import event_labels
from target_sound_detection.src.utils import median_filter, decode_with_timestamps
import clip

def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator

def initialize_model(config, ckpt, device):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt,map_location='cpu')["state_dict"], strict=False)

    model = model.to(device)
    model.cond_stage_model.to(model.device)
    model.cond_stage_model.device = model.device
    sampler = DDIMSampler(model)
    return sampler

def initialize_model_inpaint(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt,map_location='cpu')["state_dict"], strict=False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    print(model.device,device,model.cond_stage_model.device)
    sampler = DDIMSampler(model)
    return sampler
def select_best_audio(prompt,wav_list):
    clap_model = CLAPWrapper('text_to_audio/Make_An_Audio/useful_ckpts/CLAP/CLAP_weights_2022.pth','text_to_audio/Make_An_Audio/useful_ckpts/CLAP/config.yml',use_cuda=torch.cuda.is_available())
    text_embeddings = clap_model.get_text_embeddings([prompt])
    score_list = []
    for data in wav_list:
        sr,wav = data
        audio_embeddings = clap_model.get_audio_embeddings([(torch.FloatTensor(wav),sr)], resample=True)
        score = clap_model.compute_similarity(audio_embeddings, text_embeddings,use_logit_scale=False).squeeze().cpu().numpy()
        score_list.append(score)
    max_index = np.array(score_list).argmax()
    print(score_list,max_index)
    return wav_list[max_index]


class T2I:
    def __init__(self, device):
        print("Initializing T2I to %s" % device)
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        self.text_refine_tokenizer = AutoTokenizer.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")
        self.text_refine_model = AutoModelForCausalLM.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")
        self.text_refine_gpt2_pipe = pipeline("text-generation", model=self.text_refine_model, tokenizer=self.text_refine_tokenizer, device=self.device)
        self.pipe.to(device)

    @prompts(name="Generate Image From User Input Text",
             description="useful when you want to generate an image from a user input text and save it to a file. "
                         "like: generate an image of an object or something, or generate an image that includes some objects. "
                         "The input to this tool should be a string, representing the text used to generate image. ")

    def inference(self, text):
        image_filename = os.path.join('image', str(uuid.uuid4())[0:8] + ".png")
        refined_text = self.text_refine_gpt2_pipe(text)[0]["generated_text"]
        print(f'{text} refined to {refined_text}')
        image = self.pipe(refined_text).images[0]
        image.save(image_filename)
        print(f"Processed T2I.run, text: {text}, image_filename: {image_filename}")
        return image_filename

class ImageCaptioning:
    def __init__(self, device):
        print("Initializing ImageCaptioning to %s" % device)
        self.device = device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)


    @prompts(name="Remove Something From The Photo",
             description="useful when you want to remove and object or something from the photo "
                         "from its description or location. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the object need to be removed. ")

    def inference(self, image_path):
        inputs = self.processor(Image.open(image_path), return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
        captions = self.processor.decode(out[0], skip_special_tokens=True)
        return captions

class T2A:
    def __init__(self, device):
        print("Initializing Make-An-Audio to %s" % device)
        self.device = device
        self.sampler = initialize_model('text_to_audio/Make_An_Audio/configs/text-to-audio/txt2audio_args.yaml', 'text_to_audio/Make_An_Audio/useful_ckpts/ta40multi_epoch=000085.ckpt', device=device) 
        self.vocoder = VocoderBigVGAN('text_to_audio/Make_An_Audio/vocoder/logs/bigv16k53w',device=device)


    def txt2audio(self, text, seed = 55, scale = 1.5, ddim_steps = 100, n_samples = 3, W = 624, H = 80):
        SAMPLE_RATE = 16000
        prng = np.random.RandomState(seed)
        start_code = prng.randn(n_samples, self.sampler.model.first_stage_model.embed_dim, H // 8, W // 8)
        start_code = torch.from_numpy(start_code).to(device=self.device, dtype=torch.float32)
        uc = self.sampler.model.get_learned_conditioning(n_samples * [""])
        c = self.sampler.model.get_learned_conditioning(n_samples * [text])
        shape = [self.sampler.model.first_stage_model.embed_dim, H//8, W//8]  # (z_dim, 80//2^x, 848//2^x)
        samples_ddim, _ = self.sampler.sample(S = ddim_steps,
                                            conditioning = c,
                                            batch_size = n_samples,
                                            shape = shape,
                                            verbose = False,
                                            unconditional_guidance_scale = scale,
                                            unconditional_conditioning = uc,
                                            x_T = start_code)

        x_samples_ddim = self.sampler.model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0) # [0, 1]

        wav_list = []
        for idx,spec in enumerate(x_samples_ddim):
            wav = self.vocoder.vocode(spec)
            wav_list.append((SAMPLE_RATE,wav))
        best_wav = select_best_audio(text, wav_list)
        return best_wav

    @prompts(name="Generate Audio From User Input Text",
             description="useful for when you want to generate an audio "
                         "from a user input text and it saved it to a file."
                         "The input to this tool should be a string, "
                         "representing the text used to generate audio.")
    
    def inference(self, text, seed = 55, scale = 1.5, ddim_steps = 100, n_samples = 3, W = 624, H = 80):
        melbins,mel_len = 80,624
        with torch.no_grad():
            result = self.txt2audio(
                text = text,
                H = melbins,
                W = mel_len
            )
        audio_filename = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
        soundfile.write(audio_filename, result[1], samplerate = 16000)
        print(f"Processed T2I.run, text: {text}, audio_filename: {audio_filename}")
        return audio_filename

class I2A:
    def __init__(self, device):
        print("Initializing Make-An-Audio-Image to %s" % device)
        self.device = device
        self.sampler = initialize_model('text_to_audio/Make_An_Audio/configs/img_to_audio/img2audio_args.yaml', 'text_to_audio/Make_An_Audio/useful_ckpts/ta54_epoch=000216.ckpt', device=device)
        self.vocoder = VocoderBigVGAN('text_to_audio/Make_An_Audio/vocoder/logs/bigv16k53w',device=device)


    def img2audio(self, image, seed = 55, scale = 3, ddim_steps = 100, W = 624, H = 80):
        SAMPLE_RATE = 16000
        n_samples = 1 # only support 1 sample
        prng = np.random.RandomState(seed)
        start_code = prng.randn(n_samples, self.sampler.model.first_stage_model.embed_dim, H // 8, W // 8)
        start_code = torch.from_numpy(start_code).to(device=self.device, dtype=torch.float32)
        uc = self.sampler.model.get_learned_conditioning(n_samples * [""])
        #image = Image.fromarray(image)
        image = Image.open(image)
        image = self.sampler.model.cond_stage_model.preprocess(image).unsqueeze(0)
        image_embedding = self.sampler.model.cond_stage_model.forward_img(image)
        c = image_embedding.repeat(n_samples, 1, 1)
        shape = [self.sampler.model.first_stage_model.embed_dim, H//8, W//8]  # (z_dim, 80//2^x, 848//2^x)
        samples_ddim, _ = self.sampler.sample(S=ddim_steps,
                                            conditioning=c,
                                            batch_size=n_samples,
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc,
                                            x_T=start_code)

        x_samples_ddim = self.sampler.model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0) # [0, 1]
        wav_list = []
        for idx,spec in enumerate(x_samples_ddim):
            wav = self.vocoder.vocode(spec)
            wav_list.append((SAMPLE_RATE,wav))
        best_wav = wav_list[0]
        return best_wav

    @prompts(name="Generate Audio From The Image",
             description="useful for when you want to generate an audio "
                         "based on an image. "
                         "The input to this tool should be a string, "
                         "representing the image_path. ")
    
    def inference(self, image, seed = 55, scale = 3, ddim_steps = 100, W = 624, H = 80):
        melbins,mel_len = 80,624
        with torch.no_grad():
            result = self.img2audio(
                image=image,
                H=melbins, 
                W=mel_len
            )
        audio_filename = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
        soundfile.write(audio_filename, result[1], samplerate = 16000)
        print(f"Processed I2a.run, image_filename: {image}, audio_filename: {audio_filename}")
        return audio_filename

class TTS:
    def __init__(self, device=None):
        self.model = TTSInference(device)
    
    @prompts(name="Synthesize Speech Given the User Input Text",
             description="useful for when you want to convert a user input text into speech audio it saved it to a file."
                         "The input to this tool should be a string, "
                         "representing the text used to be converted to speech.")

    def inference(self, text):
        inp = {"text": text}
        out = self.model.infer_once(inp)
        audio_filename = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
        soundfile.write(audio_filename, out, samplerate = 22050)
        return audio_filename

class T2S:
    def __init__(self, device= None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Initializing DiffSinger to %s" % device)
        self.device = device
        self.exp_name = 'checkpoints/0831_opencpop_ds1000'
        self.config= 'NeuralSeq/egs/egs_bases/svs/midi/e2e/opencpop/ds1000.yaml'
        self.set_model_hparams()
        self.pipe = DiffSingerE2EInfer(self.hp, device)
        self.default_inp = {
            'text': '你 说 你 不 SP 懂 为 何 在 这 时 牵 手 AP',
            'notes': 'D#4/Eb4 | D#4/Eb4 | D#4/Eb4 | D#4/Eb4 | rest | D#4/Eb4 | D4 | D4 | D4 | D#4/Eb4 | F4 | D#4/Eb4 | D4 | rest',
            'notes_duration': '0.113740 | 0.329060 | 0.287950 | 0.133480 | 0.150900 | 0.484730 | 0.242010 | 0.180820 | 0.343570 | 0.152050 | 0.266720 | 0.280310 | 0.633300 | 0.444590'
        }


    def set_model_hparams(self):
        set_hparams(config=self.config, exp_name=self.exp_name, print_hparams=False)
        self.hp = hp

    @prompts(name="Generate Singing Voice From User Input Text, Note and Duration Sequence",
             description="useful for when you want to generate a piece of singing voice (Optional: from User Input Text, Note and Duration Sequence) "
                         "and save it to a file."
                         "If Like: Generate a piece of singing voice, the input to this tool should be \"\" since there is no User Input Text, Note and Duration Sequence. "
                         "If Like: Generate a piece of singing voice. Text: xxx, Note: xxx, Duration: xxx. "
                         "Or Like: Generate a piece of singing voice. Text is xxx, note is xxx, duration is xxx."
                         "The input to this tool should be a comma seperated string of three, "
                         "representing text, note and duration sequence since User Input Text, Note and Duration Sequence are all provided. ")
    
    def inference(self, inputs):
        self.set_model_hparams()
        val = inputs.split(",")
        key = ['text', 'notes', 'notes_duration']
        try:
            inp = {k: v for k, v in zip(key, val)}
            wav = self.pipe.infer_once(inp)
        except:
            print('Error occurs. Generate default audio sample.\n')
            inp = self.default_inp
            wav = self.pipe.infer_once(inp)
        wav *= 32767
        audio_filename = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
        wavfile.write(audio_filename, self.hp['audio_sample_rate'], wav.astype(np.int16))
        print(f"Processed T2S.run, audio_filename: {audio_filename}")
        return audio_filename

class TTS_OOD:
    def __init__(self, device):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Initializing GenerSpeech to %s" % device)
        self.device = device
        self.exp_name = 'checkpoints/GenerSpeech'
        self.config = 'NeuralSeq/modules/GenerSpeech/config/generspeech.yaml'
        self.set_model_hparams()
        self.pipe = GenerSpeechInfer(self.hp, device)

    def set_model_hparams(self):
        set_hparams(config=self.config, exp_name=self.exp_name, print_hparams=False)
        f0_stats_fn = f'{hp["binary_data_dir"]}/train_f0s_mean_std.npy'
        if os.path.exists(f0_stats_fn):
            hp['f0_mean'], hp['f0_std'] = np.load(f0_stats_fn)
            hp['f0_mean'] = float(hp['f0_mean'])
            hp['f0_std'] = float(hp['f0_std'])
        hp['emotion_encoder_path'] = 'checkpoints/Emotion_encoder.pt'
        self.hp = hp

    @prompts(name="Style Transfer",
             description="useful for when you want to generate speech samples with styles "
                         "(e.g., timbre, emotion, and prosody) derived from a reference custom voice. "
                         "Like: Generate a speech with style transferred from this voice. The text is xxx., or speak using the voice of this audio. The text is xxx."
                         "The input to this tool should be a comma seperated string of two, "
                         "representing reference audio path and input text. " )
    
    def inference(self, inputs):
        self.set_model_hparams()
        key = ['ref_audio', 'text']
        val = inputs.split(",")
        inp = {k: v for k, v in zip(key, val)}
        wav = self.pipe.infer_once(inp)
        wav *= 32767
        audio_filename = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
        wavfile.write(audio_filename, self.hp['audio_sample_rate'], wav.astype(np.int16))
        print(
            f"Processed GenerSpeech.run. Input text:{val[1]}. Input reference audio: {val[0]}. Output Audio_filename: {audio_filename}")
        return audio_filename
    
class Inpaint:
    def __init__(self, device):
        print("Initializing Make-An-Audio-inpaint to %s" % device)
        self.device = device
        self.sampler = initialize_model_inpaint('text_to_audio/Make_An_Audio/configs/inpaint/txt2audio_args.yaml', 'text_to_audio/Make_An_Audio/useful_ckpts/inpaint7_epoch00047.ckpt')
        self.vocoder = VocoderBigVGAN('text_to_audio/Make_An_Audio/vocoder/logs/bigv16k53w',device=device)
        self.cmap_transform = matplotlib.cm.viridis

    def make_batch_sd(self, mel, mask, num_samples=1):

        mel = torch.from_numpy(mel)[None,None,...].to(dtype=torch.float32)
        mask = torch.from_numpy(mask)[None,None,...].to(dtype=torch.float32)
        masked_mel = (1 - mask) * mel

        mel = mel * 2 - 1
        mask = mask * 2 - 1
        masked_mel = masked_mel * 2 -1

        batch = {
             "mel": repeat(mel.to(device=self.device), "1 ... -> n ...", n=num_samples),
             "mask": repeat(mask.to(device=self.device), "1 ... -> n ...", n=num_samples),
             "masked_mel": repeat(masked_mel.to(device=self.device), "1 ... -> n ...", n=num_samples),
        }
        return batch
    def gen_mel(self, input_audio_path):
        SAMPLE_RATE = 16000
        sr, ori_wav = wavfile.read(input_audio_path)
        print("gen_mel")
        print(sr,ori_wav.shape,ori_wav)
        ori_wav = ori_wav.astype(np.float32, order='C') / 32768.0
        if len(ori_wav.shape)==2:# stereo
            ori_wav = librosa.to_mono(ori_wav.T)# gradio load wav shape could be (wav_len,2) but librosa expects (2,wav_len)
        print(sr,ori_wav.shape,ori_wav)
        ori_wav = librosa.resample(ori_wav,orig_sr = sr,target_sr = SAMPLE_RATE)

        mel_len,hop_size = 848,256
        input_len = mel_len * hop_size
        if len(ori_wav) < input_len:
            input_wav = np.pad(ori_wav,(0,mel_len*hop_size),constant_values=0)
        else:
            input_wav = ori_wav[:input_len]
 
        mel = TRANSFORMS_16000(input_wav)
        return mel
    def gen_mel_audio(self, input_audio):
        SAMPLE_RATE = 16000
        sr,ori_wav = input_audio
        print("gen_mel_audio")
        print(sr,ori_wav.shape,ori_wav)

        ori_wav = ori_wav.astype(np.float32, order='C') / 32768.0
        if len(ori_wav.shape)==2:# stereo
            ori_wav = librosa.to_mono(ori_wav.T)# gradio load wav shape could be (wav_len,2) but librosa expects (2,wav_len)
        print(sr,ori_wav.shape,ori_wav)
        ori_wav = librosa.resample(ori_wav,orig_sr = sr,target_sr = SAMPLE_RATE)

        mel_len,hop_size = 848,256
        input_len = mel_len * hop_size
        if len(ori_wav) < input_len:
            input_wav = np.pad(ori_wav,(0,mel_len*hop_size),constant_values=0)
        else:
            input_wav = ori_wav[:input_len]
        mel = TRANSFORMS_16000(input_wav)
        return mel
    def inpaint(self, batch, seed, ddim_steps, num_samples=1, W=512, H=512):
        model = self.sampler.model
    
        prng = np.random.RandomState(seed)
        start_code = prng.randn(num_samples, model.first_stage_model.embed_dim, H // 8, W // 8)
        start_code = torch.from_numpy(start_code).to(device=self.device, dtype=torch.float32)

        c = model.get_first_stage_encoding(model.encode_first_stage(batch["masked_mel"]))
        cc = torch.nn.functional.interpolate(batch["mask"],
                                                size=c.shape[-2:])
        c = torch.cat((c, cc), dim=1) # (b,c+1,h,w) 1 is mask

        shape = (c.shape[1]-1,)+c.shape[2:]
        samples_ddim, _ = self.sampler.sample(S=ddim_steps,
                                            conditioning=c,
                                            batch_size=c.shape[0],
                                            shape=shape,
                                            verbose=False)
        x_samples_ddim = model.decode_first_stage(samples_ddim)

    
        mask = batch["mask"]# [-1,1]
        mel = torch.clamp((batch["mel"]+1.0)/2.0,min=0.0, max=1.0)
        mask = torch.clamp((batch["mask"]+1.0)/2.0,min=0.0, max=1.0)
        predicted_mel = torch.clamp((x_samples_ddim+1.0)/2.0,min=0.0, max=1.0)
        inpainted = (1-mask)*mel+mask*predicted_mel
        inpainted = inpainted.cpu().numpy().squeeze()
        inapint_wav = self.vocoder.vocode(inpainted)

        return inpainted, inapint_wav
    def predict(self, input_audio, mel_and_mask, seed = 55, ddim_steps = 100):
        SAMPLE_RATE = 16000
        torch.set_grad_enabled(False)
        mel_img = Image.open(mel_and_mask['image'])
        mask_img = Image.open(mel_and_mask["mask"])
        show_mel = np.array(mel_img.convert("L"))/255
        mask = np.array(mask_img.convert("L"))/255
        mel_bins,mel_len = 80,848
        input_mel = self.gen_mel_audio(input_audio)[:,:mel_len]
        mask = np.pad(mask,((0,0),(0,mel_len-mask.shape[1])),mode='constant',constant_values=0)
        print(mask.shape,input_mel.shape)
        with torch.no_grad():
            batch = self.make_batch_sd(input_mel,mask,num_samples=1)
            inpainted,gen_wav = self.inpaint(
                batch=batch,
                seed=seed,
                ddim_steps=ddim_steps,
                num_samples=1,
                H=mel_bins, W=mel_len
            )
        inpainted = inpainted[:,:show_mel.shape[1]]
        color_mel = self.cmap_transform(inpainted)
        input_len = int(input_audio[1].shape[0] * SAMPLE_RATE / input_audio[0])
        gen_wav = (gen_wav * 32768).astype(np.int16)[:input_len]
        image = Image.fromarray((color_mel*255).astype(np.uint8))
        image_filename = os.path.join('image', str(uuid.uuid4())[0:8] + ".png")
        image.save(image_filename)
        audio_filename = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
        soundfile.write(audio_filename, gen_wav, samplerate = 16000)
        return image_filename, audio_filename

    @prompts(name="Audio Inpainting",
             description="useful for when you want to inpaint a mel spectrum of an audio and predict this audio, "
                         "this tool will generate a mel spectrum and you can inpaint it, receives audio_path as input. "
                         "The input to this tool should be a string, "
                         "representing the audio_path. " )
    
    def inference(self, input_audio_path):
        crop_len = 500
        crop_mel = self.gen_mel(input_audio_path)[:,:crop_len]
        color_mel = self.cmap_transform(crop_mel)
        image = Image.fromarray((color_mel*255).astype(np.uint8))
        image_filename = os.path.join('image', str(uuid.uuid4())[0:8] + ".png")
        image.save(image_filename)
        return image_filename
    
class ASR:
    def __init__(self, device):
        print("Initializing Whisper to %s" % device)
        self.device = device
        self.model = whisper.load_model("base", device=device)

    @prompts(name="Transcribe speech",
             description="useful for when you want to know the text corresponding to a human speech, "
                         "receives audio_path as input. "
                         "The input to this tool should be a string, "
                         "representing the audio_path. " )    

    def inference(self, audio_path):
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.device)
        _, probs = self.model.detect_language(mel)
        options = whisper.DecodingOptions()
        result = whisper.decode(self.model, mel, options)
        return result.text
        
    def translate_english(self, audio_path):
        audio = self.model.transcribe(audio_path, language='English')
        return audio['text']

class A2T:
    def __init__(self, device):
        print("Initializing Audio-To-Text Model to %s" % device)
        self.device = device
        self.model = AudioCapModel("audio_to_text/audiocaps_cntrstv_cnn14rnn_trm")

    @prompts(name="Generate Text From The Audio",
             description="useful for when you want to describe an audio in text, "
                         "receives audio_path as input. "
                         "The input to this tool should be a string, "
                         "representing the audio_path. " )    

    def inference(self, audio_path):
        audio = whisper.load_audio(audio_path)
        caption_text = self.model(audio)
        return caption_text[0]

class SoundDetection:
    def __init__(self, device):
        self.device = device
        self.sample_rate = 32000
        self.window_size = 1024
        self.hop_size = 320
        self.mel_bins = 64
        self.fmin = 50
        self.fmax = 14000
        self.model_type = 'PVT'
        self.checkpoint_path = 'audio_detection/audio_infer/useful_ckpts/audio_detection.pth'
        self.classes_num = detection_config.classes_num
        self.labels = detection_config.labels
        self.frames_per_second = self.sample_rate // self.hop_size
        # Model = eval(self.model_type)
        self.model = PVT(sample_rate=self.sample_rate, window_size=self.window_size, 
            hop_size=self.hop_size, mel_bins=self.mel_bins, fmin=self.fmin, fmax=self.fmax, 
            classes_num=self.classes_num)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(device)

    @prompts(name="Detect The Sound Event From The Audio",
             description="useful for when you want to know what event in the audio and the sound event start or end time, it will return an image "
                         "receives audio_path as input. "
                         "The input to this tool should be a string, "
                         "representing the audio_path. " )  
    
    def inference(self, audio_path):
        # Forward
        (waveform, _) = librosa.core.load(audio_path, sr=self.sample_rate, mono=True)
        waveform = waveform[None, :]    # (1, audio_length)
        waveform = torch.from_numpy(waveform)
        waveform = waveform.to(self.device)
        # Forward
        with torch.no_grad():
            self.model.eval()
            batch_output_dict = self.model(waveform, None)
        framewise_output = batch_output_dict['framewise_output'].data.cpu().numpy()[0]
        """(time_steps, classes_num)"""
        # print('Sound event detection result (time_steps x classes_num): {}'.format(
        #     framewise_output.shape))
        import numpy as np
        import matplotlib.pyplot as plt
        sorted_indexes = np.argsort(np.max(framewise_output, axis=0))[::-1]
        top_k = 10  # Show top results
        top_result_mat = framewise_output[:, sorted_indexes[0 : top_k]]    
        """(time_steps, top_k)"""
        # Plot result    
        stft = librosa.core.stft(y=waveform[0].data.cpu().numpy(), n_fft=self.window_size, 
            hop_length=self.hop_size, window='hann', center=True)
        frames_num = stft.shape[-1]
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
        axs[0].matshow(np.log(np.abs(stft)), origin='lower', aspect='auto', cmap='jet')
        axs[0].set_ylabel('Frequency bins')
        axs[0].set_title('Log spectrogram')
        axs[1].matshow(top_result_mat.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
        axs[1].xaxis.set_ticks(np.arange(0, frames_num, self.frames_per_second))
        axs[1].xaxis.set_ticklabels(np.arange(0, frames_num / self.frames_per_second))
        axs[1].yaxis.set_ticks(np.arange(0, top_k))
        axs[1].yaxis.set_ticklabels(np.array(self.labels)[sorted_indexes[0 : top_k]])
        axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
        axs[1].set_xlabel('Seconds')
        axs[1].xaxis.set_ticks_position('bottom')
        plt.tight_layout()
        image_filename = os.path.join('image', str(uuid.uuid4())[0:8] + ".png")
        plt.savefig(image_filename)
        return image_filename

class SoundExtraction:
    def __init__(self, device):
        self.device = device
        self.model_file = 'sound_extraction/useful_ckpts/LASSNet.pt'
        self.stft = STFT()
        import torch.nn as nn
        self.model = nn.DataParallel(LASSNet(device)).to(device)
        checkpoint = torch.load(self.model_file)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

    @prompts(name="Extract Sound Event From Mixture Audio Based On Language Description",
             description="useful for when you extract target sound from a mixture audio, you can describe the target sound by text, "
                         "receives audio_path and text as input. "
                         "The input to this tool should be a comma seperated string of two, "
                         "representing mixture audio path and input text." ) 
    
    def inference(self, inputs):
        #key = ['ref_audio', 'text']
        val = inputs.split(",")
        audio_path = val[0] # audio_path, text
        text = val[1]
        waveform = load_wav(audio_path)
        waveform = torch.tensor(waveform).transpose(1,0)
        mixed_mag, mixed_phase = self.stft.transform(waveform)
        text_query = ['[CLS] ' + text]
        mixed_mag = mixed_mag.transpose(2,1).unsqueeze(0).to(self.device)
        est_mask = self.model(mixed_mag, text_query)
        est_mag = est_mask * mixed_mag  
        est_mag = est_mag.squeeze(1)  
        est_mag = est_mag.permute(0, 2, 1) 
        est_wav = self.stft.inverse(est_mag.cpu().detach(), mixed_phase)
        est_wav = est_wav.squeeze(0).squeeze(0).numpy()  
        #est_path = f'output/est{i}.wav'
        audio_filename = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
        print('audio_filename ', audio_filename)
        save_wav(est_wav, audio_filename)
        return audio_filename


class Binaural:
    def __init__(self, device):
        self.device = device
        self.model_file = 'mono2binaural/useful_ckpts/m2b/binaural_network.net'
        self.position_file = ['mono2binaural/useful_ckpts/m2b/tx_positions.txt',
                              'mono2binaural/useful_ckpts/m2b/tx_positions2.txt',
                              'mono2binaural/useful_ckpts/m2b/tx_positions3.txt',
                              'mono2binaural/useful_ckpts/m2b/tx_positions4.txt',
                              'mono2binaural/useful_ckpts/m2b/tx_positions5.txt']
        self.net = BinauralNetwork(view_dim=7,
                      warpnet_layers=4,
                      warpnet_channels=64,
                      )
        self.net.load_from_file(self.model_file)
        self.sr = 48000

    @prompts(name="Sythesize Binaural Audio From A Mono Audio Input",
             description="useful for when you want to transfer your mono audio into binaural audio, "
                         "receives audio_path as input. "
                         "The input to this tool should be a string, "
                         "representing the audio_path. " ) 
    
    def inference(self, audio_path):
        mono, sr  = librosa.load(path=audio_path, sr=self.sr, mono=True)
        mono = torch.from_numpy(mono)
        mono = mono.unsqueeze(0)
        import numpy as np
        import random
        rand_int = random.randint(0,4)
        view = np.loadtxt(self.position_file[rand_int]).transpose().astype(np.float32)
        view = torch.from_numpy(view)
        if not view.shape[-1] * 400 == mono.shape[-1]:
            mono = mono[:,:(mono.shape[-1]//400)*400] # 
            if view.shape[1]*400 > mono.shape[1]:
                m_a = view.shape[1] - mono.shape[-1]//400 
                rand_st = random.randint(0,m_a)
                view = view[:,m_a:m_a+(mono.shape[-1]//400)] # 
        # binauralize and save output
        self.net.eval().to(self.device)
        mono, view = mono.to(self.device), view.to(self.device)
        chunk_size = 48000  # forward in chunks of 1s
        rec_field =  1000  # add 1000 samples as "safe bet" since warping has undefined rec. field
        rec_field -= rec_field % 400  # make sure rec_field is a multiple of 400 to match audio and view frequencies
        chunks = [
            {
                "mono": mono[:, max(0, i-rec_field):i+chunk_size],
                "view": view[:, max(0, i-rec_field)//400:(i+chunk_size)//400]
            }
            for i in range(0, mono.shape[-1], chunk_size)
        ]
        for i, chunk in enumerate(chunks):
            with torch.no_grad():
                mono = chunk["mono"].unsqueeze(0)
                view = chunk["view"].unsqueeze(0)
                binaural = self.net(mono, view).squeeze(0)
                if i > 0:
                    binaural = binaural[:, -(mono.shape[-1]-rec_field):]
                chunk["binaural"] = binaural
        binaural = torch.cat([chunk["binaural"] for chunk in chunks], dim=-1)
        binaural = torch.clamp(binaural, min=-1, max=1).cpu()
        #binaural = chunked_forwarding(net, mono, view)
        audio_filename = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
        import torchaudio
        torchaudio.save(audio_filename, binaural, sr)
        #soundfile.write(audio_filename, binaural, samplerate = 48000)
        print(f"Processed Binaural.run, audio_filename: {audio_filename}")
        return audio_filename

class TargetSoundDetection:
    def __init__(self, device):
        self.device = device
        self.MEL_ARGS = {
            'n_mels': 64,
            'n_fft': 2048,
            'hop_length': int(22050 * 20 / 1000),
            'win_length': int(22050 * 40 / 1000)
        }
        self.EPS = np.spacing(1)
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.event_labels = event_labels
        self.id_to_event =  {i : label for i, label in enumerate(self.event_labels)}
        config = torch.load('audio_detection/target_sound_detection/useful_ckpts/tsd/run_config.pth', map_location='cpu')
        config_parameters = dict(config)
        config_parameters['tao'] = 0.6
        if 'thres' not in config_parameters.keys():
            config_parameters['thres'] = 0.5
        if 'time_resolution' not in config_parameters.keys():
            config_parameters['time_resolution'] = 125
        model_parameters = torch.load('audio_detection/target_sound_detection/useful_ckpts/tsd/run_model_7_loss=-0.0724.pt'
                                        , map_location=lambda storage, loc: storage) # load parameter 
        self.model = getattr(tsd_models, config_parameters['model'])(config_parameters,
                    inputdim=64, outputdim=2, time_resolution=config_parameters['time_resolution'], **config_parameters['model_args'])
        self.model.load_state_dict(model_parameters)
        self.model = self.model.to(self.device).eval()
        self.re_embeds = torch.load('audio_detection/target_sound_detection/useful_ckpts/tsd/text_emb.pth')
        self.ref_mel = torch.load('audio_detection/target_sound_detection/useful_ckpts/tsd/ref_mel.pth')

    def extract_feature(self, fname):
        import soundfile as sf
        y, sr = sf.read(fname, dtype='float32')
        print('y ', y.shape)
        ti = y.shape[0]/sr
        if y.ndim > 1:
            y = y.mean(1)
        y = librosa.resample(y, sr, 22050)
        lms_feature = np.log(librosa.feature.melspectrogram(y, **self.MEL_ARGS) + self.EPS).T
        return lms_feature,ti
    
    def build_clip(self, text):
        text = clip.tokenize(text).to(self.device) # ["a diagram with dog", "a dog", "a cat"]
        text_features = self.clip_model.encode_text(text)
        return text_features
    
    def cal_similarity(self, target, retrievals):
        ans = []
        for name in retrievals.keys():
            tmp = retrievals[name]
            s = torch.cosine_similarity(target.squeeze(), tmp.squeeze(), dim=0)
            ans.append(s.item())
        return ans.index(max(ans))

    @prompts(name="Target Sound Detection",
             description="useful for when you want to know when the target sound event in the audio happens. You can use language descriptions to instruct the model， "
                         "receives text description and audio_path as input. "
                         "The input to this tool should be a comma seperated string of two, "
                         "representing audio path and the text description. " ) 
    
    def inference(self, inputs):
        audio_path, text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        target_emb = self.build_clip(text) # torch type
        idx = self.cal_similarity(target_emb, self.re_embeds)
        target_event = self.id_to_event[idx]
        embedding = self.ref_mel[target_event]
        embedding = torch.from_numpy(embedding)
        embedding = embedding.unsqueeze(0).to(self.device).float()
        inputs,ti = self.extract_feature(audio_path)
        inputs = torch.from_numpy(inputs)
        inputs = inputs.unsqueeze(0).to(self.device).float()
        decision, decision_up, logit = self.model(inputs, embedding)
        pred = decision_up.detach().cpu().numpy()
        pred = pred[:,:,0]
        frame_num = decision_up.shape[1]
        time_ratio = ti / frame_num
        filtered_pred = median_filter(pred, window_size=1, threshold=0.5)
        time_predictions = []
        for index_k in range(filtered_pred.shape[0]):
            decoded_pred = []
            decoded_pred_ = decode_with_timestamps(target_event, filtered_pred[index_k,:])
            if len(decoded_pred_) == 0: # neg deal
                decoded_pred_.append((target_event, 0, 0))
            decoded_pred.append(decoded_pred_)
            for num_batch in range(len(decoded_pred)): # when we test our model,the batch_size is 1
                cur_pred = pred[num_batch]
                # Save each frame output, for later visualization
                label_prediction = decoded_pred[num_batch] # frame predict
                for event_label, onset, offset in label_prediction:
                    time_predictions.append({
                        'onset': onset*time_ratio,
                        'offset': offset*time_ratio,})
        ans = ''
        for i,item in enumerate(time_predictions):
            ans = ans + 'segment' + str(i+1) + ' start_time: ' + str(item['onset']) + '  end_time: ' + str(item['offset']) + '\t'
        return ans

class Speech_Enh_SC:
    """Speech Enhancement or Separation in single-channel
    Example usage:
        enh_model = Speech_Enh_SS("cuda")
        enh_wav = enh_model.inference("./test_chime4_audio_M05_440C0213_PED_REAL.wav")
    """
    def __init__(self, device="cuda", model_name="espnet/Wangyou_Zhang_chime4_enh_train_enh_conv_tasnet_raw"):
        self.model_name = model_name
        self.device = device
        print("Initializing ESPnet Enh to %s" % device)
        self._initialize_model()

    def _initialize_model(self):
        from espnet_model_zoo.downloader import ModelDownloader
        from espnet2.bin.enh_inference import SeparateSpeech

        d = ModelDownloader()

        cfg = d.download_and_unpack(self.model_name)
        self.separate_speech = SeparateSpeech(
            train_config=cfg["train_config"],
            model_file=cfg["model_file"],
            # for segment-wise process on long speech
            segment_size=2.4,
            hop_size=0.8,
            normalize_segment_scale=False,
            show_progressbar=True,
            ref_channel=None,
            normalize_output_wav=True,
            device=self.device,
        )
        
    @prompts(name="Speech Enhancement In Single-Channel",
             description="useful for when you want to enhance the quality of the speech signal by reducing background noise (single-channel), "
                         "receives audio_path as input."
                         "The input to this tool should be a string, "
                         "representing the audio_path. " ) 
    
    def inference(self, speech_path, ref_channel=0):
        speech, sr = soundfile.read(speech_path)
        speech = speech[:, ref_channel]
        enh_speech = self.separate_speech(speech[None, ...], fs=sr)
        audio_filename = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
        soundfile.write(audio_filename, enh_speech[0].squeeze(), samplerate=sr)
        return audio_filename

class Speech_SS:
    def __init__(self, device="cuda", model_name="lichenda/wsj0_2mix_skim_noncausal"):
        self.model_name = model_name
        self.device = device
        print("Initializing ESPnet SS to %s" % device)
        self._initialize_model()

    def _initialize_model(self):
        from espnet_model_zoo.downloader import ModelDownloader
        from espnet2.bin.enh_inference import SeparateSpeech

        d = ModelDownloader()

        cfg = d.download_and_unpack(self.model_name)
        self.separate_speech = SeparateSpeech(
            train_config=cfg["train_config"],
            model_file=cfg["model_file"],
            # for segment-wise process on long speech
            segment_size=2.4,
            hop_size=0.8,
            normalize_segment_scale=False,
            show_progressbar=True,
            ref_channel=None,
            normalize_output_wav=True,
            device=self.device,
        )

    @prompts(name="Speech Separation",
             description="useful for when you want to separate each speech from the speech mixture, "
                         "receives audio_path as input."
                         "The input to this tool should be a string, "
                         "representing the audio_path. " ) 
    
    def inference(self, speech_path):
        speech, sr = soundfile.read(speech_path)
        enh_speech = self.separate_speech(speech[None, ...], fs=sr)
        audio_filename = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
        if len(enh_speech) == 1:
            soundfile.write(audio_filename, enh_speech[0].squeeze(), samplerate=sr)
        else:
            audio_filename_1 = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
            soundfile.write(audio_filename_1, enh_speech[0].squeeze(), samplerate=sr)
            audio_filename_2 = os.path.join('audio', str(uuid.uuid4())[0:8] + ".wav")
            soundfile.write(audio_filename_2, enh_speech[1].squeeze(), samplerate=sr)
            audio_filename = merge_audio(audio_filename_1, audio_filename_2)
        return audio_filename