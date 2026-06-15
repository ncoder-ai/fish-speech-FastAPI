#!/usr/bin/env python3
"""
Long-scene tests: multi-voice dialogue + inline emotion tags, 400-500 words each.
Saves generated audio to /mnt/truenas_public/fish_speech_tests and validates it.

Usage:
  .venv/bin/python tools/test_long_scenes.py --base-url http://localhost:8770
"""
import argparse
import os
import subprocess
import time

import httpx

OUT_DIR = "/mnt/truenas_public/fish_speech_tests"

# --- Scene A: multi-voice dialogue, 3 speakers (~460 words) -------------------
SCENE_MULTIVOICE = (
    "<|speaker:0|>Welcome back to the evening broadcast. Tonight we are joined by two "
    "guests who could not disagree more about the future of our city. Let us begin. "
    "<|speaker:1|>Thank you for having me. I will say it plainly: the new transit plan "
    "is the most important thing this council has done in a decade. It connects the "
    "neighborhoods that have been ignored for far too long, and it does so without "
    "raising a single cent in property taxes. "
    "<|speaker:2|>That is a lovely story, but it is just that, a story. The numbers do "
    "not add up. You are promising trains that run every six minutes on a budget that "
    "can barely keep the buses running today. Where exactly is the money coming from? "
    "<|speaker:1|>From the federal grant we secured in March, and from the savings we "
    "get by retiring the oldest vehicles in the fleet. I have the figures right here. "
    "<|speaker:2|>Figures on paper. I have seen a dozen plans like this collapse the "
    "moment the first winter storm hits and the maintenance bills come due. "
    "<|speaker:0|>Let me bring this back to the people watching at home. If I am a "
    "commuter standing in the cold tomorrow morning, what changes for me? "
    "<|speaker:1|>Within a year, your wait is cut in half. Within three years, you can "
    "cross the entire city in under forty minutes. That is real time given back to "
    "families, to workers, to everyone. "
    "<|speaker:2|>Or within three years you are standing on a half built platform "
    "wondering why the council spent your money on a promise it could not keep. "
    "<|speaker:0|>Let us talk about timing, because that is where these plans usually "
    "fall apart. When does the first line actually open to the public? "
    "<|speaker:1|>The eastern line opens in eighteen months. We have already broken "
    "ground at three stations, and the tunneling crews are ahead of schedule for the "
    "first time in this city's history. "
    "<|speaker:2|>Ahead of schedule today, perhaps. But you and I both know the hardest "
    "stretch runs straight under the river, and no one has priced in what happens if "
    "the ground there is softer than the surveys claim. "
    "<|speaker:1|>The surveys were independent, peer reviewed, and paid for out of the "
    "contingency budget precisely so that nobody could accuse us of cutting corners. "
    "<|speaker:2|>Then I look forward to reading them in full, line by line, because "
    "the people deserve to know exactly what they are paying for. "
    "<|speaker:0|>Strong words from both sides. We will be following this closely as "
    "the vote approaches. Thank you both for a spirited and honest debate tonight, and "
    "thank you all for listening. We will see you again same time tomorrow."
)

# --- Scene B: single narrator, rich emotion tags (~440 words) -----------------
SCENE_EMOTION = (
    "[excited] You will not believe what happened to me today. I have to tell you "
    "everything, right from the very beginning. [pause] So I woke up late, already "
    "running behind, and I was absolutely certain the whole day was ruined. "
    "[sigh] I missed the early train, I spilled coffee on my favorite shirt, and the "
    "rain started the moment I stepped outside. [sad] Honestly, for a few minutes "
    "there, I just wanted to turn around and go straight back to bed. "
    "[whisper] But then something strange happened. As I stood under the awning, "
    "soaked and miserable, an old woman tapped me on the shoulder. "
    "[surprised] She handed me an envelope with my name written on the front, and "
    "before I could ask a single question, she simply smiled and walked away into the "
    "crowd. [pause] Inside the envelope was a letter from my grandmother, written "
    "years ago, that she had asked a friend to deliver on this exact day. "
    "[moved] I read it standing there in the rain, and I could not stop the tears. "
    "She wrote that she was proud of me, that she had always known I would find my "
    "way, even on the hardest mornings. [crying] I must have read it ten times. "
    "[laughing] And then, can you believe it, the sun came out, right on cue, like "
    "the whole sky was in on the joke. [excited] Suddenly the missed train did not "
    "matter, the ruined shirt did not matter, none of it mattered at all. "
    "[shouting] I actually laughed out loud in the middle of the street like a "
    "complete lunatic. [low voice] So here is what I learned today, and I want you to "
    "really hear me on this. The worst mornings sometimes carry the best gifts, "
    "hidden just out of sight, waiting for you to stop and notice them. "
    "[pause] And there is more, because the day was not finished with me yet. "
    "[excited] When I finally got to the office, two hours late and looking like a "
    "drowned cat, my whole team was standing by the door waiting for me. "
    "[surprised] I thought I was in trouble, I really did, my heart just sank right "
    "through the floor. [laughing] But they had a cake, and balloons, and a ridiculous "
    "hand drawn banner, because somehow they all remembered it was the anniversary of "
    "my very first day. [moved] These people, who I sometimes complain about over "
    "dinner, had planned the whole thing in secret for a week. "
    "[sigh] I felt so foolish for ever doubting that this was where I belonged. "
    "[whisper] So if you take one thing from this rambling story of mine, let it be "
    "this. [emphasis] Pay attention to the small kindnesses. They are everywhere, "
    "every single day, and they are the whole point of all of it."
)

# --- Scene C: multi-voice + emotion combined (~450 words) ---------------------
SCENE_COMBINED = (
    "<|speaker:0|>[whisper] Keep your voice down. They said the house was empty, but I "
    "heard something move upstairs. "
    "<|speaker:1|>[nervous] I told you we should not have come here at night. This "
    "place gives me the creeps. Let us just grab the box and go. "
    "<|speaker:0|>[low voice] Quiet. Listen. [pause] There it is again, right above us. "
    "<|speaker:1|>[shocked] Okay, that is definitely not the wind. That is footsteps. "
    "<|speaker:0|>[excited] Wait, the box is right here, under the stairs, exactly "
    "where the letter said it would be. Help me lift it. "
    "<|speaker:1|>[panting] It is heavier than it looks. What on earth did your uncle "
    "leave in here? "
    "<|speaker:0|>[surprised] I have no idea, but it is locked, and there is a note "
    "taped to the top. [pause] It says, for the brave ones who finally came home. "
    "<|speaker:1|>[whisper] Your uncle had a strange sense of humor. Open it. "
    "<|speaker:0|>[delight] It is full of old photographs, and letters, and, look at "
    "this, the deed to the lake house. He left it all to us. "
    "<|speaker:1|>[laughing] After all that sneaking around in the dark, terrified out "
    "of our minds, it was a gift the whole time. "
    "<|speaker:0|>[moved] He always said this family forgot how to be together. Maybe "
    "this was his way of bringing us back. "
    "<|speaker:1|>[sigh] Come on. Let us get out of this dusty old place and go see "
    "that lake before the sun comes up. "
    "<|speaker:0|>[excited] Wait, there is one more thing in the bottom of the box. A "
    "key, an old brass key, with a paper tag tied to it. "
    "<|speaker:1|>[curious] What does the tag say? Read it. "
    "<|speaker:0|>[whisper] It says, the boathouse, midnight, do not be late. "
    "<|speaker:1|>[shocked] Midnight? That is in twenty minutes. Your uncle planned "
    "this down to the very hour, did he not? "
    "<|speaker:0|>[laughing] Of course he did. He never could resist a bit of theatre. "
    "Come on, if we run we can just make it. "
    "<|speaker:1|>[panting] Slow down, the path is covered in roots and I cannot see a "
    "thing out here. "
    "<|speaker:0|>[excited] There it is, the boathouse, and look, there is a light on "
    "inside. Someone is waiting for us. "
    "<|speaker:1|>[nervous] After everything tonight, I am not sure my heart can take "
    "another surprise. "
    "<|speaker:0|>[moved] It is the whole family, every last one of them, sitting "
    "around a table with candles. They came. They actually all came. "
    "<|speaker:1|>[crying] He did it. From wherever he is now, the old man finally got "
    "us all in one room again. "
    "<|speaker:0|>[warm] Then let us not waste a second of it. Come on, they are "
    "waving us in."
)

# --- Scene D: extra-long mixed scene (~700+ spoken words) --------------------
# Deliberately past the old ~550-650 word ceiling to prove the sliding context
# window keeps long-form generation alive instead of raising "Prompt too long".
_EXTRA_TURNS = [
    "<|speaker:0|>[excited] Captain, we have cleared the upper atmosphere and the "
    "engines are holding steady. All systems are green across the board.",
    "<|speaker:1|>[calm] Good. Steady as she goes. Bring the main array online and "
    "give me a reading on the fuel reserves before we commit to the burn.",
    "<|speaker:2|>[nervous] Reserves are at sixty percent, but I am seeing a flicker "
    "in the third coolant line. It could be nothing, or it could be the start of "
    "something we really do not want.",
    "<|speaker:1|>[firm] Watch it closely and report any change immediately. We did "
    "not come this far to lose the ship to a bad valve.",
    "<|speaker:0|>[surprised] Captain, I am picking up a signal. It is faint, but it "
    "is structured. Someone, or something, is broadcasting on the old emergency band.",
    "<|speaker:1|>[low voice] Put it on the speakers. Let us hear what we are dealing "
    "with before we decide our next move.",
    "<|speaker:2|>[whisper] It is repeating the same three tones, over and over. That "
    "is the distress pattern from the lost colony ships. They were supposed to be gone "
    "for thirty years.",
    "<|speaker:0|>[moved] If there are survivors out here, after all this time, we are "
    "the only ones close enough to reach them.",
    "<|speaker:1|>[determined] Then the decision makes itself. Plot a course to the "
    "source of that signal. We burn on my mark.",
    "<|speaker:2|>[panting] Course laid in. Coolant line is holding for now. I am "
    "transferring all non essential power to the engines.",
    "<|speaker:0|>[excited] Approaching the coordinates now. There is a ship out "
    "there, drifting, dark, but the hull is intact. I am reading faint life signs.",
    "<|speaker:1|>[relieved] They are alive. After everything, they are actually "
    "alive. Open a channel and tell them help has finally arrived.",
    "<|speaker:2|>[crying] Thirty years of silence, and we are the ones who answer. I "
    "never thought I would see a day like this.",
    "<|speaker:1|>[warm] Easy now. Steady hands. Let us bring them home together, the "
    "way we always said we would.",
    "<|speaker:0|>[laughing] Docking clamps engaged. Hatch is sealed and pressurized. "
    "They are coming aboard, Captain. They are really coming aboard.",
]
SCENE_EXTRALONG = " ".join(_EXTRA_TURNS * 2)

SCENES = {
    "multivoice_dialogue": SCENE_MULTIVOICE,
    "emotion_narration": SCENE_EMOTION,
    "multivoice_emotion_combined": SCENE_COMBINED,
    "extralong_stress": SCENE_EXTRALONG,
}


def decoded_seconds(data: bytes) -> float:
    try:
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", "44100", "pipe:1"],
            input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
        )
        return len(p.stdout) / 2 / 44100
    except Exception:
        return -1


def run_scene(base, name, text, fmt):
    words = len(text.replace("|", " ").split())
    payload = {"input": text, "response_format": fmt, "stream": False,
               "temperature": 0.8, "top_p": 0.8}
    t = time.time()
    try:
        r = httpx.post(f"{base}/v1/audio/speech", json=payload, timeout=600)
    except Exception as e:
        print(f"  FAIL {name}.{fmt}: request error {e}")
        return False
    dt = time.time() - t
    if r.status_code != 200:
        print(f"  FAIL {name}.{fmt}: status={r.status_code} {r.text[:300]}")
        return False
    path = os.path.join(OUT_DIR, f"{name}.{fmt}")
    with open(path, "wb") as f:
        f.write(r.content)
    dur = decoded_seconds(r.content)
    if dur > 5.0:
        print(f"  PASS {name}.{fmt}: ~{words} words -> {dur:.1f}s audio, "
              f"{len(r.content)} bytes in {dt:.1f}s -> {path}")
        return True
    print(f"  FAIL {name}.{fmt}: bad audio dur={dur} bytes={len(r.content)} -> {path}")
    return False


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8770")
    ap.add_argument("--formats", default="wav,mp3")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--only", default="",
                    help="comma-separated scene names to run (default: all)")
    args = ap.parse_args()
    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)
    fmts = [f.strip() for f in args.formats.split(",") if f.strip()]
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    npass, nfail = 0, 0
    for name, text in SCENES.items():
        if only and name not in only:
            continue
        print(f"== {name} ==")
        for fmt in fmts:
            if run_scene(args.base_url, name, text, fmt):
                npass += 1
            else:
                nfail += 1
    print(f"\n==== long scenes: {npass} passed, {nfail} failed ====")
    print(f"saved to {OUT_DIR}")
    raise SystemExit(1 if nfail else 0)


if __name__ == "__main__":
    main()
