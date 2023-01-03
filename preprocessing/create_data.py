#!/usr/env/bin/python3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from os.path import join
import numpy as np
import soundfile as sf
import glob
import argparse
import time
import json
from tqdm import tqdm
import shutil
import scipy.signal as ss
import io 
import scipy.io.wavfile 
import pyroomacoustics as pra

from utils import obtain_noise_file

SEED = 100
np.random.seed(SEED)

bwe_params = {
	"scale_factors": [2, 4, 8],
	"scale_probas": [.33, .33, .34],
	"lp_types": ["bessel", "butter", "cheby2"],
	"lp_orders": [2, 4, 8]
}

enh_params = {
	"snr_range": [0, 20]
}

derev_params = {
	"t60_range": [0.4, 1.0],
	"dim_range": [5, 15, 5, 15, 2, 6],
	"min_distance_to_wall": 1.
}

ROOT = "" ## put your root directory here
assert ROOT != "", "You need to have a root databases directory"

parser = argparse.ArgumentParser()

parser.add_argument('--task', type=str, choices=["enh", "derev", "derev+enh", "bwe"])
parser.add_argument('--speech', type=str, choices=["vctk", "wsj0", "dns"], default="wsj0", help='Clean speech')
parser.add_argument('--noise', type=str, choices=["none", "chime", "qut", "wham"], default="chime", help='Noise')
parser.add_argument('--sr', type=int, default=16000)
parser.add_argument('--splits', type=str, default="valid,train,test", help='Split folders of the dataset')
parser.add_argument('--corruption-per-sample', type=int, default=1)
parser.add_argument('--dummy', action="store_true", help='Number of samples')
parser.add_argument('--bwe-method', type=str, default="polyphase", choices=["decimate", "polyphase"])

args = parser.parse_args()
splits = args.splits.strip().split(",")

params = vars(args)
if "enh" in args.task:
	params["noise_dirs"] = {
		"wham": {split:f"/data/lemercier/databases/whamr/wham_noise/{split}" for split in splits},
		"chime": {split:f"/data/lemercier/databases/CHiME3/data/audio/16kHz/backgrounds" for split in splits},
		"qut": {split:f"/data/lemercier/ databases/dns_chime3/wav16k/min/{split}" for split in splits}
	}
	params = {**enh_params, **params}
if "derev" in args.task:
	params = {**derev_params, **params}
if "bwe" in args.task:
	params = {**bwe_params, **params}

output_dir = join(ROOT, args.speech + "_" + args.task + "_git_test")

t0 = time.time()

if args.speech == "wsj0":
	dic_split = {"valid": "si_dt_05", "train": "si_tr_s", "test": "si_et_05"}
	speech_lists = {split:glob.glob(f"{ROOT}/WSJ0/wsj0/{dic_split[split]}/**/*.wav") for split in splits}
elif args.speech == "vctk":
	speakers = sorted(os.listdir(f"{ROOT}/VCTK-Corpus/wav48/"))
	speakers.remove("p280")
	speakers.remove("p315")
	ranges = {"train": [0, 99], "valid": [97, 99], "test": [99, 107]}
	speech_lists  = {split:[] for split in splits}
	for split in splits:
		for spk_idx in range(*ranges[split]):
			speech_lists[split] += glob.glob(f"{ROOT}/VCTK-Corpus/wav48/{speakers[spk_idx]}/*.wav")

if os.path.exists(output_dir):
	shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)
log = open(join(output_dir, "log_stats.txt"), "w")
log.write("Parameters \n ========== \n")
for key, param in params.items():
	log.write(key + " : " + str(param) + "\n")

for i_split, split in enumerate(splits):

	print("Processing split n° {}: {}...".format(i_split+1, split))
	
	clean_output_dir = join(output_dir, split, "clean")
	noisy_output_dir = join(output_dir, split, "noisy")

	os.makedirs(clean_output_dir, exist_ok=True)
	os.makedirs(noisy_output_dir, exist_ok=True)

	speech_list = speech_lists[split]
	speech_dir = None
	real_nb_samples = 5 if args.dummy else len(speech_list)
	nb_corruptions_per_sample = 1 if split == "test" else args.corruption_per_sample

	for i_sample in tqdm(range(real_nb_samples)):

		speech_basename = os.path.basename(speech_list[i_sample])
		speech, sr = sf.read(speech_list[i_sample])
		assert sr == args.sr, "Obtained an unexpected Sampling rate"
		original_scale = np.max(np.abs(speech))

		for ic in range(nb_corruptions_per_sample):

			lossy_speech = speech.copy()




			### Dereverberation
			if "derev" in args.task:

				t60 = np.random.uniform(params["t60_range"][0], params["t60_range"][1]) #sample t60
				room_dim = np.array([ np.random.uniform(params["dim_range"][2*n], params["dim_range"][2*n+1]) for n in range(3) ]) #sample Dimensions
				center_mic_position = np.array([ np.random.uniform(params["min_distance_to_wall"], room_dim[n] - params["min_distance_to_wall"]) for n in range(3) ]) #sample microphone position
				source_position = np.array([ np.random.uniform(params["min_distance_to_wall"], room_dim[n] - params["min_distance_to_wall"]) for n in range(3) ]) #sample source position
				distance_source = 1/np.sqrt(center_mic_position.ndim)*np.linalg.norm(center_mic_position - source_position)
				mic_array_2d = pra.beamforming.circular_2D_array(center_mic_position[: -1], 1, phi0=0, radius=1.) # Compute microphone array
				mic_array = np.pad(mic_array_2d, ((0, 1), (0, 0)), mode="constant", constant_values=center_mic_position[-1])

				### Reverberant Room
				e_absorption, max_order = pra.inverse_sabine(t60, room_dim) #Compute absorption coeff
				reverberant_room = pra.ShoeBox(
					room_dim, fs=16000, materials=pra.Material(e_absorption), max_order=min(3, max_order), ray_tracing=True
				) # Create room
				reverberant_room.set_ray_tracing()

				reverberant_room.add_microphone_array(mic_array) # Add microphone array

				# Generate reverberant room
				reverberant_room.add_source(source_position, signal=lossy_speech)
				reverberant_room.compute_rir()
				reverberant_room.simulate()
				t60_real = np.mean(reverberant_room.measure_rt60()).squeeze()
				lossy_speech = np.squeeze(np.array(reverberant_room.mic_array.signals))

				#compute target
				e_absorption_dry = 0.99
				dry_room = pra.ShoeBox(
					room_dim, fs=16000, materials=pra.Material(e_absorption_dry), max_order=0
				) # Create room
				dry_room.add_microphone_array(mic_array) # Add microphone array

				# Generate dry room
				dry_room.add_source(source_position, signal=speech) 
				dry_room.compute_rir()
				dry_room.simulate()
				t60_real_dry = np.mean(dry_room.measure_rt60()).squeeze()
				speech = np.squeeze(np.array(dry_room.mic_array.signals))
				noise_floor_snr = 50
				noise_floor_power = 1/speech.shape[0]*np.sum(speech**2)*np.power(10,-noise_floor_snr/10)
				noise_floor_signal = np.random.rand(int(.5*args.sr)) * np.sqrt(noise_floor_power)
				speech = np.concatenate([ speech, noise_floor_signal ])
				
				min_length = min(lossy_speech.shape[0], speech.shape[0])
				lossy_speech, speech = lossy_speech[: min_length], speech[: min_length]





			### Enhancement

			if "enh" in args.task:

				noise, noise_sr = obtain_noise_file(params["noise_dirs"][args.noise][split], i_sample, 1, dataset=args.noise, sample_rate=args.sr, len_speech=speech.shape[0])
				noise = np.squeeze(noise)
				if noise.shape[0] < speech.shape[0]:
					noise = np.pad(noise, ((0, speech.shape[0] - noise.shape[0])))
				else:
					noise = noise[: speech.shape[0]]
					
				snr = np.random.uniform(params["snr_range"][0], params["snr_range"][1])
				noise_power = 1/noise.shape[0]*np.sum(noise**2)
				speech_power = 1/speech.shape[0]*np.sum(speech**2)
				noise_power_target = speech_power*np.power(10,-snr/10)
				noise_scaling = np.sqrt(noise_power_target / noise_power)
				if "derev" in args.task:
					lossy_speech = lossy_speech + noise_scaling * noise
				else:
					lossy_speech = speech + noise_scaling * noise





			### Bandwidth Reduction

			if "bwe" in args.task:
				scale_factor = np.random.choice(params["scale_factors"], p=params["scale_probas"])
				lp_type = np.random.choice(params["lp_types"])
				lp_order = np.random.choice(params["lp_orders"])
				Wn = 1./(2*scale_factor)
				if lp_type == "bessel":
					kwargs = {}
				elif lp_type == "butter":
					kwargs = {}
				elif lp_type == "cheby2": 
					kwargs = {"rs": 10. + 20*np.random.random()}

				if lp_order > 2:
					kwargs["output"] = "sos"
				lp_filter_coefs = getattr(scipy.signal, lp_type)(N=lp_order, Wn=Wn, fs=1, **kwargs)

				if args.bwe_method == "decimate": #method used by HiFI++ and VoiceFixer
					z, p, k = ss.sos2zpk(lp_filter_coefs) if lp_order > 2 else ss.tf2zpk(*lp_filter_coefs)
					filter_instance = ss.dlti(z,p,k)
					lossy_speech_subsampled = ss.decimate(lossy_speech, q=scale_factor, ftype=filter_instance)
					lossy_speech = ss.resample_poly(lossy_speech_subsampled, up=scale_factor, down=1)
				elif args.bwe_method == "polyphase": #method used by NVSR
					sos = lp_filter_coefs if lp_order > 2 else ss.tf2sos(*lp_filter_coefs)
					lossy_speech_filtered = ss.sosfilt(sos, lossy_speech)
					lossy_speech_subsampled = ss.resample_poly(lossy_speech_filtered, down=scale_factor, up=1)
					lossy_speech = ss.resample_poly(lossy_speech_subsampled, up=args.sr, down=sr/scale_factor)



			filename =  f"{i_sample*args.corruption_per_sample + ic}_" + speech_basename[: -4]
			if "derev" in args.task:
				filename += f"_t60={t60_real:.2f}"
			if "enh" in args.task:
				filename += f"_snr={snr:.1f}"
			if "bwe" in args.task:
				filename += f"_down={scale_factor}"
			filename += ".wav"

			### Export
			sf.write(join(clean_output_dir, filename), speech, args.sr)
			sf.write(join(noisy_output_dir, filename), lossy_speech, args.sr)