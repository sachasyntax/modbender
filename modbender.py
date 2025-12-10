#!/usr/bin/env python3
import argparse
import wave
import numpy as np
import random
from collections import defaultdict
import sys

# IO
def load_wav_mono_channels(path):
    try:
        w = wave.open(path, 'rb')
    except wave.Error:
        print("Error: input file is not a valid WAV")
        sys.exit(1)
    params = w.getparams()
    nch, sw, rate = params.nchannels, params.sampwidth, params.framerate
    frames = w.readframes(params.nframes)
    w.close()
    if sw != 2:
        print("Error: only WAV PCM 16-bit supported")
        sys.exit(1)
    audio = np.frombuffer(frames, dtype=np.int16)
    channels = []
    if nch == 1:
        channels.append(audio.copy())
    else:
        channels.append(audio[0::2].copy())
        channels.append(audio[1::2].copy())
    return channels, rate

def write_wav_from_channels(path, channels, rate):
    nch = len(channels)
    minlen = min(len(ch) for ch in channels)
    chans = [ch[:minlen] for ch in channels]
    if nch == 1:
        out = chans[0]
    else:
        inter = np.empty(minlen * nch, dtype=np.int16)
        inter[0::2] = chans[0]
        inter[1::2] = chans[1]
        out = inter
    w = wave.open(path, 'wb')
    w.setnchannels(nch)
    w.setsampwidth(2)
    w.setframerate(rate)
    w.writeframes(out.tobytes())
    w.close()

# markov 
def build_markov_nth_order_patterns(patterns, order=3):
    transitions = defaultdict(list)
    if len(patterns) <= order:
        return transitions
    for i in range(len(patterns) - order):
        key = tuple(patterns[i:i+order])
        transitions[key].append(patterns[i+order])
    return transitions

def generate_markov_sequence_patterns(patterns, transitions, length, order=3):
    if length <= 0:
        return []
    if len(patterns) < order:
        return [random.choice(patterns) for _ in range(length)]
    idx = random.randint(0, len(patterns)-order)
    current = tuple(patterns[idx:idx+order])
    out = list(current)
    for _ in range(length - order):
        if current in transitions and transitions[current]:
            nxt = random.choice(transitions[current])
        else:
            nxt = random.choice(patterns)
        out.append(nxt)
        current = tuple(list(current[1:]) + [nxt])
    return out[:length]

# LSB 
def lsb_random(channel_samples):
    mask = np.random.randint(0, 2, size=channel_samples.shape, dtype=np.uint16)
    return (channel_samples.astype(np.uint16) ^ mask).astype(np.int16)

def lsb_markov(channel_samples):
    bits = (channel_samples.astype(np.uint16) & 1).astype(np.uint8)
    if len(bits) < 2:
        return channel_samples
    transitions = {0: [], 1: []}
    for i in range(len(bits)-1):
        transitions[int(bits[i])].append(int(bits[i+1]))
    current = int(bits[random.randint(0, len(bits)-2)])
    new_bits = [current]
    for _ in range(len(bits)-1):
        options = transitions.get(current)
        nxt = random.choice(options) if options else random.randint(0,1)
        new_bits.append(nxt)
        current = nxt
    new_bits = np.array(new_bits, dtype=np.uint16)
    cleared = (channel_samples.astype(np.uint16) & 0xFFFE)
    return (cleared | new_bits).astype(np.int16)

# XOR 
def apply_xor_markov(channel_samples, patterns, xor_seq):
    out = []
    pos = 0
    for w, patt in zip(xor_seq, patterns):
        chunk = channel_samples[pos:pos+w]
        if len(chunk) == 0:
            break
        out.append(np.bitwise_xor(chunk.astype(np.uint16),
                                  np.full_like(chunk, patt, dtype=np.uint16)))
        pos += w
    if pos < len(channel_samples):
        remaining = channel_samples[pos:]
        out.append(np.bitwise_xor(remaining.astype(np.uint16),
                                  np.full_like(remaining, patterns[0], dtype=np.uint16)))
    return np.concatenate(out).astype(np.int16)

# markov on samples
def frame_audio_variable(samples, windows):
    chunks = []
    pos = 0
    for w in windows:
        chunk = samples[pos:pos+w]
        if len(chunk) == 0:
            break
        chunks.append(chunk)
        pos += w
    return chunks

def build_markov_window_transitions(chunks, order=2):
    transitions = defaultdict(list)
    if len(chunks) <= order:
        return transitions
    for i in range(len(chunks)-order):
        key = tuple(chunks[i+j].tobytes() for j in range(order))
        transitions[key].append(chunks[i+order])
    return transitions

def generate_markov_window_reorder(chunks, transitions, order=2):
    if len(chunks) == 0:
        return chunks
    if len(chunks) <= order:
        return np.concatenate(chunks)
    idx = random.randint(0, len(chunks)-order-1)
    current = tuple(chunks[idx + j] for j in range(order))
    out = list(current)
    for _ in range(len(chunks)-order):
        key = tuple(c.tobytes() for c in current)
        nxt = random.choice(transitions[key]) if key in transitions and transitions[key] else random.choice(chunks)
        out.append(nxt)
        current = tuple(list(current[1:]) + [nxt])
    return np.concatenate(out)

# bytewise markov
def frame_bytes_variable(samples, bwindows):
    chunks = []
    pos = 0
    for w in bwindows:
        chunk = samples[pos:pos+w]
        if len(chunk)==0:
            break
        chunks.append(chunk)
        pos += w
    return chunks

def build_markov_byte_transitions(chunks, order=2):
    transitions = defaultdict(list)
    if len(chunks) <= order:
        return transitions
    for i in range(len(chunks)-order):
        key = tuple(chunks[i+j] for j in range(order))
        transitions[key].append(chunks[i+order])
    return transitions

def generate_markov_byte_reorder(chunks, transitions, order=2):
    if len(chunks)==0:
        return chunks
    if len(chunks)<=order:
        return b''.join(chunks)
    idx = random.randint(0,len(chunks)-order-1)
    current = tuple(chunks[idx + j] for j in range(order))
    out = list(current)
    for _ in range(len(chunks)-order):
        key = tuple(current)
        nxt = random.choice(transitions[key]) if key in transitions and transitions[key] else random.choice(chunks)
        out.append(nxt)
        current = tuple(list(current[1:]) + [nxt])
    return b''.join(out)

# mapping samples ampitude / noise on window sizes
def envelope(samples, window=1024):
    extra = len(samples) % window
    trimmed = samples[:-extra] if extra else samples
    chunks = trimmed.reshape(-1, window)
    env = np.mean(np.abs(chunks), axis=1)
    env /= np.max(env) + 1e-9
    return np.repeat(env, window)[:len(samples)]

def local_noise(samples, window=1024):
    extra = len(samples) % window
    trimmed = samples[:-extra] if extra else samples
    chunks = trimmed.reshape(-1, window)
    noise = np.std(chunks, axis=1)
    noise /= np.max(noise) + 1e-9
    return np.repeat(noise, window)[:len(samples)]

def map_continuous_window(env, w_min, w_max, smooth=0.1, gain=5):
    windows = []
    prev_w = (w_min + w_max)//2
    for val in env:
        s = 1 / (1 + np.exp(-gain*(val-0.5)))
        target = w_min + (w_max - w_min) * s
        w = int(prev_w*(1-smooth) + target*smooth)
        windows.append(w)
        prev_w = w
    return np.array(windows)

# pipeline 
def run_pipeline_on_channel(channel, cfg, mwindows=None, xwindows=None, bwindows=None):
    s = channel.copy()
    for step in cfg['order']:
        if step == 'l' and cfg['lsb_on']:
            s = lsb_random(s) if cfg['lsb_mode']=='random' else lsb_markov(s)

        elif step == 'x' and cfg['xor_on']:
            patterns = cfg['xor_patterns']
            seq = xwindows if xwindows is not None else [random.choice(patterns)]*len(s)
            s = apply_xor_markov(s, patterns, seq)

        elif step == 'm' and cfg['markov_reorder']:
            chunks = frame_audio_variable(
                s,
                mwindows if mwindows is not None
                else [cfg['mwindow']]*((len(s)+cfg['mwindow']-1)//cfg['mwindow'])
            )
            transitions = build_markov_window_transitions(chunks, order=cfg['markovorder'])
            s = generate_markov_window_reorder(chunks, transitions, order=cfg['markovorder'])

        elif step == 'b' and cfg['bmarkov']:
            raw_bytes = s.tobytes()
            chunks = frame_bytes_variable(
                raw_bytes,
                bwindows if bwindows is not None
                else [cfg['bwindow']]*((len(raw_bytes)+cfg['bwindow']-1)//cfg['bwindow'])
            )
            transitions = build_markov_byte_transitions(chunks, order=cfg['bmarkovorder'])
            new_bytes = generate_markov_byte_reorder(chunks, transitions, order=cfg['bmarkovorder'])
            s = np.frombuffer(new_bytes, dtype=np.int16)

    return s

# execution
def main():
    parser = argparse.ArgumentParser(description="Modular audio processor: LSB / XOR / Markov reorder / Byte-level Markov")
    parser.add_argument("input", help="File WAV input (PCM 16-bit)")
    parser.add_argument("output", help="File WAV output")

    parser.add_argument("--lsb", nargs="+", default=["n"],
                        help='Either: "--lsb n" or "--lsb y random" or "--lsb y markov"')

    parser.add_argument("--xor", choices=["y","n"], default="y")
    parser.add_argument("--patterns", nargs="+", type=lambda x: int(x,0))

    parser.add_argument("--markov", choices=["y","n"], default="n")
    parser.add_argument("--markovorder", type=int, default=3)

    parser.add_argument("--bmarkov", choices=["y","n"], default="n")
    parser.add_argument("--bmarkovorder", type=int, default=3, help="byte-level markov order")

    parser.add_argument("--bwindow", type=int, default=512)
    parser.add_argument("--mwindow", type=int, default=None)
    parser.add_argument("--xorwindow", type=int, default=None)

    parser.add_argument("--rate", type=int, default=None)
    parser.add_argument("--iterations", "-n", type=int, default=1)

    parser.add_argument("--order", nargs=4, metavar=('A','B','C','D'),
                        choices=['l','x','m','b'],
                        default=['b','l','x','m'])

    parser.add_argument("--smooth", type=float, default=0.1)

    args = parser.parse_args()

    # LSB
    if args.lsb[0]=='y':
        if len(args.lsb)<2:
            print("Error: '--lsb y' needs to specify 'random' o 'markov'")
            sys.exit(1)
        lsb_on = True
        lsb_mode = args.lsb[1].lower()
        if lsb_mode not in ('random','markov'):
            print("Errore: not valid LSB mode")
            sys.exit(1)
    else:
        lsb_on = False
        lsb_mode = None

    xor_on = args.xor=='y'
    markov_reorder = args.markov=='y'
    bmarkov = args.bmarkov=='y'

    channels, in_rate = load_wav_mono_channels(args.input)
    out_rate = args.rate if args.rate else in_rate

    cfg = {
        'lsb_on': lsb_on,
        'lsb_mode': lsb_mode,
        'xor_on': xor_on,
        'xor_patterns': args.patterns if args.patterns else [0x0001,0x00FF,0x0F0F,0xAAAA],
        'markov_reorder': markov_reorder,
        'bmarkov': bmarkov,
        'bwindow': args.bwindow,
        'mwindow': args.mwindow or 512,
        'xorwindow': args.xorwindow or 512,
        'order': args.order,
        'markovorder': args.markovorder,
        'bmarkovorder': args.bmarkovorder,
        'smooth': args.smooth
    }

    processed_channels = []
    for ch in channels:
        env = envelope(ch, window=1024)
        mw_seq = map_continuous_window(env, w_min=32, w_max=3072, smooth=args.smooth)
        xw_seq = map_continuous_window(env, w_min=32, w_max=3072, smooth=args.smooth)
        bw_seq = None  # fixed bytewise markov windows

        ch_work = ch.copy()
        for _ in range(args.iterations):
            ch_work = run_pipeline_on_channel(
                ch_work,
                cfg,
                mwindows=mw_seq,
                xwindows=xw_seq,
                bwindows=bw_seq
            )
        processed_channels.append(ch_work)

    write_wav_from_channels(args.output, processed_channels, out_rate)
    print(f"Wrote output: {args.output} (samplerate {out_rate} Hz)")

if __name__=="__main__":
    main()
