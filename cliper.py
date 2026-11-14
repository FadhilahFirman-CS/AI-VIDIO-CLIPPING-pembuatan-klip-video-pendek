#!/usr/bin/env python3
"""
Video Clipper Engine
--------------------
Otomatis memotong video panjang (misal 10 menit) menjadi ringkasan 20-30 detik
yang hanya berisi bagian paling "inti".

Cara kerja (skor per jendela waktu):
  1. Energi audio   -> bagian yang lantang / ada aksi suara
  2. Aktivitas suara-> bagian yang ada orang bicara (bukan hening)
  3. Perubahan scene-> bagian yang visualnya banyak berubah
  4. (opsional) Transkrip Whisper -> bagian dengan kata kunci / kalimat padat

Segmen skor tertinggi dipilih, dibatasi total durasi target, lalu
digabung berurutan sesuai alur waktu asli dengan crossfade singkat.

Kebutuhan: ffmpeg, ffprobe, Python 3.9+, numpy
Opsional : pip install faster-whisper   (untuk skor berbasis ucapan)

Contoh:
  python video_clipper.py input.mp4 -o hasil             # 4 video: hasil_1.mp4 ... hasil_4.mp4
  python video_clipper.py input.mp4 --clips 3 --target 30
  python video_clipper.py input.mp4 --vertical           # format 9:16 (Reels/TikTok/Shorts)
  python video_clipper.py input.mp4 --merge              # satu video gabungan (mode lama)
  python video_clipper.py input.mp4 --whisper            # pakai transkrip ucapan
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np


# ----------------------------------------------------------------- util ----
def run(cmd, capture=True):
    r = subprocess.run(cmd, capture_output=capture, text=True)
    if r.returncode != 0:
        sys.exit(f"Perintah gagal:\n{' '.join(cmd)}\n{r.stderr[-1500:]}")
    return r


def probe_duration(path):
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path])
    return float(json.loads(r.stdout)["format"]["duration"])


def has_audio(path):
    r = run(["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "json", path])
    return bool(json.loads(r.stdout).get("streams"))


def norm(x):
    """Normalisasi 0..1 (robust terhadap outlier)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


# ------------------------------------------------------------- analisis ----
def audio_features(path, n_win, win, tmpdir):
    """Energi RMS per jendela dan rasio 'bersuara' (non-hening)."""
    wav = os.path.join(tmpdir, "a.wav")
    sr = 16000
    run(["ffmpeg", "-y", "-i", path, "-vn", "-ac", "1", "-ar", str(sr),
         "-f", "wav", wav])
    raw = np.frombuffer(open(wav, "rb").read()[44:], dtype=np.int16)
    raw = raw.astype(np.float32) / 32768.0
    per = int(sr * win)
    energy = np.zeros(n_win)
    voiced = np.zeros(n_win)
    for i in range(n_win):
        chunk = raw[i * per:(i + 1) * per]
        if chunk.size == 0:
            continue
        rms = np.sqrt(np.mean(chunk ** 2))
        energy[i] = rms
        # frame 25ms: berapa persen frame di atas ambang hening
        fl = int(sr * 0.025)
        frames = chunk[:len(chunk) // fl * fl].reshape(-1, fl)
        fr = np.sqrt((frames ** 2).mean(axis=1))
        voiced[i] = (fr > 0.01).mean()
    return energy, voiced


def scene_features(path, n_win, win):
    """Jumlah perubahan scene per jendela (ffmpeg scene detection)."""
    r = subprocess.run(
        ["ffmpeg", "-i", path, "-vf", "select='gt(scene,0.25)',showinfo",
         "-an", "-f", "null", "-"], capture_output=True, text=True)
    times = [float(m) for m in re.findall(r"pts_time:([0-9.]+)", r.stderr)]
    counts = np.zeros(n_win)
    for t in times:
        idx = int(t // win)
        if 0 <= idx < n_win:
            counts[idx] += 1
    return counts


def transcript_features(path, n_win, win, tmpdir):
    """Skor kepadatan ucapan + kata kunci via faster-whisper (opsional)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("[!] faster-whisper belum terpasang, lewati skor transkrip "
              "(pip install faster-whisper)")
        return None
    wav = os.path.join(tmpdir, "w.wav")
    run(["ffmpeg", "-y", "-i", path, "-vn", "-ac", "1", "-ar", "16000", wav])
    model = WhisperModel("base", compute_type="int8")
    segs, _ = model.transcribe(wav, vad_filter=True)
    words = np.zeros(n_win)
    kw = np.zeros(n_win)
    # kata penanda poin penting (Indonesia + Inggris)
    keywords = {"penting", "intinya", "kesimpulan", "ingat", "kunci",
                "rahasia", "pertama", "kedua", "ketiga", "ternyata",
                "important", "key", "remember", "summary", "secret",
                "first", "second", "finally", "conclusion"}
    for s in segs:
        toks = re.findall(r"\w+", s.text.lower())
        if not toks:
            continue
        a, b = int(s.start // win), int(min(s.end, n_win * win - 1e-3) // win)
        for i in range(a, b + 1):
            if 0 <= i < n_win:
                words[i] += len(toks) / (b - a + 1)
                kw[i] += sum(t in keywords for t in toks)
    return norm(words) * 0.6 + norm(kw) * 0.4


# ------------------------------------------------------------- seleksi ----
def pick_segments(score, win, seg_len, target, min_gap, count=None):
    """Pilih segmen non-overlap skor tertinggi.
    count=None -> sampai total ~ target detik (mode gabung).
    count=N    -> N segmen terbaik (mode klip terpisah)."""
    n = len(score)
    k = max(1, int(round(seg_len / win)))          # jendela per segmen
    if n < k:
        return [(0.0, n * win)]
    # skor rata-rata geser
    kernel = np.ones(k) / k
    avg = np.convolve(score, kernel, mode="valid")  # panjang n-k+1
    need = count if count else max(1, int(round(target / seg_len)))
    chosen = []
    avg = avg.copy()
    # beri sedikit bonus pada intro agar ada konteks pembuka
    avg[0] *= 1.15
    gap_w = int(round(min_gap / win))
    for _ in range(need):
        i = int(np.argmax(avg))
        if avg[i] <= -1:      # tidak ada ruang lagi
            break
        chosen.append(i)
        lo, hi = max(0, i - k - gap_w), min(len(avg), i + k + gap_w)
        avg[lo:hi] = -2
    chosen.sort()
    return [(i * win, i * win + k * win) for i in chosen]


# ----------------------------------------------------------------- render ----
def render(path, segments, out, fade, vertical, has_a, title=None):
    n = len(segments)
    vf_extra = ""
    if vertical:
        # crop tengah ke 9:16 lalu scale 1080x1920
        vf_extra = ",crop='min(iw,ih*9/16)':ih,scale=1080:1920,setsar=1"
    parts = []
    for i, (s, e) in enumerate(segments):
        parts.append(f"[0:v]trim={s:.3f}:{e:.3f},setpts=PTS-STARTPTS{vf_extra}[v{i}]")
        if has_a:
            parts.append(
                f"[0:a]atrim={s:.3f}:{e:.3f},asetpts=PTS-STARTPTS,"
                f"afade=t=in:d={fade},afade=t=out:st={max(0, e - s - fade):.3f}:d={fade}[a{i}]")
    v_in = "".join(f"[v{i}]" for i in range(n))
    if has_a:
        a_in = "".join(f"[v{i}][a{i}]" for i in range(n))
        parts.append(f"{a_in}concat=n={n}:v=1:a=1[v][a]")
        maps = ["-map", "[v]", "-map", "[a]"]
    else:
        parts.append(f"{v_in}concat=n={n}:v=1:a=0[v]")
        maps = ["-map", "[v]"]
    fc = ";".join(parts)
    cmd = ["ffmpeg", "-y", "-i", path, "-filter_complex", fc, *maps,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
           "-pix_fmt", "yuv420p"]
    if has_a:
        cmd += ["-c:a", "aac", "-b:a", "160k"]
    if title:
        cmd += ["-metadata", f"title={title}"]
    cmd += ["-movflags", "+faststart", out]
    run(cmd)


def render_single(path, start, end, out, vertical, has_a, fade, title=None):
    """Render satu klip utuh (kontinu) dari start..end."""
    vf = "setsar=1"
    if vertical:
        vf = "crop='min(iw,ih*9/16)':ih,scale=1080:1920,setsar=1"
    dur = end - start
    cmd = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", path,
           "-vf", vf]
    if has_a:
        cmd += ["-af", f"afade=t=in:d={fade},afade=t=out:st={max(0, dur - fade):.3f}:d={fade}"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p"]
    if has_a:
        cmd += ["-c:a", "aac", "-b:a", "160k"]
    if title:
        cmd += ["-metadata", f"title={title}"]
    cmd += ["-movflags", "+faststart", out]
    run(cmd)


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser(description="Auto-clip video ke ringkasan inti.")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default=None,
                    help="nama dasar output; klip jadi NAMA_1.mp4, NAMA_2.mp4, ...")
    ap.add_argument("--clips", type=int, default=4,
                    help="jumlah video terpisah yang dihasilkan, default 4")
    ap.add_argument("--merge", action="store_true",
                    help="gabung semua potongan jadi SATU video (mode lama)")
    ap.add_argument("--target", type=float, default=25,
                    help="durasi tiap klip (detik), default 25")
    ap.add_argument("--seg", type=float, default=4,
                    help="panjang potongan di mode --merge (detik), default 4")
    ap.add_argument("--gap", type=float, default=8, help="jarak minimum antar klip (detik)")
    ap.add_argument("--fade", type=float, default=0.15, help="fade audio di sambungan (detik)")
    ap.add_argument("--whisper", action="store_true", help="pakai transkrip (butuh faster-whisper)")
    ap.add_argument("--vertical", action="store_true", help="output 9:16 untuk Reels/TikTok/Shorts")
    ap.add_argument("--json", action="store_true", help="simpan daftar segmen ke .json")
    ap.add_argument("--titles", action="store_true",
                    help="simpan judul sumber sebagai metadata pada video output")
    a = ap.parse_args()

    if not os.path.isfile(a.input):
        sys.exit(f"File tidak ditemukan: {a.input}")
    base = a.output or os.path.splitext(a.input)[0] + "_clip"
    if base.lower().endswith(".mp4"):
        base = base[:-4]
    out = base + ".mp4"

    dur = probe_duration(a.input)
    if dur <= a.target:
        sys.exit(f"Video ({dur:.1f}s) sudah lebih pendek dari target ({a.target}s).")
    win = 1.0
    n_win = int(dur // win)
    audio = has_audio(a.input)
    print(f"[1/4] Menganalisis video {dur/60:.1f} menit ...")

    with tempfile.TemporaryDirectory() as tmp:
        scene = norm(scene_features(a.input, n_win, win))
        if audio:
            energy, voiced = audio_features(a.input, n_win, win, tmp)
            energy, voiced = norm(energy), norm(voiced)
        else:
            energy = voiced = np.zeros(n_win)
        tr = transcript_features(a.input, n_win, win, tmp) if (a.whisper and audio) else None

    # bobot gabungan
    if tr is not None:
        score = 0.30 * energy + 0.15 * voiced + 0.20 * scene + 0.35 * tr
    elif audio:
        score = 0.45 * energy + 0.25 * voiced + 0.30 * scene
    else:
        score = scene
    # haluskan agar tidak memilih lonjakan sesaat
    score = np.convolve(score, np.ones(3) / 3, mode="same")

    print("[2/4] Memilih bagian inti ...")
    if a.merge:
        segs = pick_segments(score, win, a.seg, a.target, a.gap)
    else:
        segs = pick_segments(score, win, a.target, a.target, a.gap, count=a.clips)
        # urutkan dari skor tertinggi supaya klip_1 = yang paling inti
        k = max(1, int(round(a.target / win)))
        segs.sort(key=lambda se: -score[int(se[0] // win):int(se[0] // win) + k].mean())
    total = sum(e - s for s, e in segs)
    for s, e in segs:
        print(f"      {s:7.1f}s -> {e:7.1f}s")
    print(f"      total {total:.1f}s dari {len(segs)} potongan")
    if not a.merge and len(segs) < a.clips:
        print(f"[!] Hanya {len(segs)} klip yang muat (diminta {a.clips}). "
              f"Kecilkan --target atau --gap untuk mendapat lebih banyak.")

    print("[3/4] Merender ...")
    outputs = []
    source_title = os.path.splitext(os.path.basename(a.input))[0]
    if a.merge:
        render(a.input, segs, out, a.fade, a.vertical, audio,
               source_title if a.titles else None)
        outputs.append(out)
    else:
        for n, (s, e) in enumerate(segs, 1):
            o = f"{base}_{n}.mp4"
            title = f"{source_title} - Clip {n}" if a.titles else None
            render_single(a.input, s, e, o, a.vertical, audio, a.fade, title)
            outputs.append(o)

    if a.json:
        with open(base + ".json", "w") as f:
            json.dump([{"start": s, "end": e} for s, e in segs], f, indent=2)

    print("[4/4] Selesai:")
    for o in outputs:
        print(f"      {o}  ({probe_duration(o):.1f}s)")


if __name__ == "__main__":
    main()
