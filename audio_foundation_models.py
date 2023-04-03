import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'NeuralSeq'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'text_to_audio/Make_An_Audio'))
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
    clap_model = CLAPWrapper('useful_ckpts/CLAP/CLAP_weights_2022.pth','useful_ckpts/CLAP/config.yml',use_cuda=torch.cuda.is_available())
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
        c = image_embedding.repeat(n_samples, 1, 1)# shape:[1,77,1280],即还没有变成句子embedding，仍是每个单词的embedding
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
        self.inferencer = TTSInference(device)
    
    def inference(self, text):
        global temp_audio_filename
        inp = {"text": text}
        out = self.inferencer.infer_once(inp)
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

    def inference(self, inputs):
        self.set_model_hparams()
        val = inputs.split(",")
        key = ['text', 'notes', 'notes_duration']
        if inputs == '' or len(val) < len(key):
            inp = self.default_inp
        else:
            inp = {k:v for k,v in zip(key,val)}
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

    def inference(self, inputs):
        self.set_model_hparams()
        key = ['ref_audio', 'text']
        val = inputs.split(",")
        inp = {k: v for k, v in zip(key, val)}
        print(inp)
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
        self.vocoder = VocoderBigVGAN('./vocoder/logs/bigv16k53w',device=device)
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
        ori_wav = ori_wav.astype(np.float32, order='C') / 32768.0 # order='C'是以C语言格式存储，不用管
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

        ori_wav = ori_wav.astype(np.float32, order='C') / 32768.0 # order='C'是以C语言格式存储，不用管
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
    def show_mel_fn(self, input_audio_path):
        crop_len = 500 # the full mel cannot be showed due to gradio's Image bug when using tool='sketch'
        crop_mel = self.gen_mel(input_audio_path)[:,:crop_len]
        color_mel = self.cmap_transform(crop_mel)
        image = Image.fromarray((color_mel*255).astype(np.uint8))
        image_filename = os.path.join('image', str(uuid.uuid4())[0:8] + ".png")
        image.save(image_filename)
        return image_filename
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
    def inference(self, input_audio, mel_and_mask, seed = 55, ddim_steps = 100):
        SAMPLE_RATE = 16000
        torch.set_grad_enabled(False)
        mel_img = Image.open(mel_and_mask['image'])
        mask_img = Image.open(mel_and_mask["mask"])
        show_mel = np.array(mel_img.convert("L"))/255 # 由于展示的mel只展示了一部分，所以需要重新从音频生成mel
        mask = np.array(mask_img.convert("L"))/255
        mel_bins,mel_len = 80,848
        input_mel = self.gen_mel_audio(input_audio)[:,:mel_len]# 由于展示的mel只展示了一部分，所以需要重新从音频生成mel
        mask = np.pad(mask,((0,0),(0,mel_len-mask.shape[1])),mode='constant',constant_values=0)# 将mask填充到原来的mel的大小
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
    
class ASR:
    def __init__(self, device):
        print("Initializing Whisper to %s" % device)
        self.device = device
        self.model = whisper.load_model("base", device=device)
    def inference(self, audio_path):
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.device)
        _, probs = self.model.detect_language(mel)
        options = whisper.DecodingOptions()
        result = whisper.decode(self.model, mel, options)
        return result.text

class A2T:
    def __init__(self, device):
        print("Initializing Audio-To-Text Model to %s" % device)
        self.device = device
        self.model = AudioCapModel("audio_to_text/audiocaps_cntrstv_cnn14rnn_trm")
    def inference(self, audio_path):
        audio = whisper.load_audio(audio_path)
        caption_text = self.model(audio)
        return caption_text[0]