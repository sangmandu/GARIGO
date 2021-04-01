import wave
import moviepy.editor as mpe

import os

def make_audio(input_route, output_route, video_name):

    ROOT_DIR = os.getcwd()
    path = os.path.join(ROOT_DIR, 'audio/')
    audio_route = os.path.join(path, "tmp_audio.mp3")
    
    clip = mpe.VideoFileClip(input_route)
    clip.audio.write_audiofile(audio_route)

    video = mpe.VideoFileClip(output_route)
    audio = clip.audio
    # final = video.set_audio(audio)
    video.audio = audio

    path = os.path.join(ROOT_DIR, 'final_video/')
    final_route = os.path.join(path, video_name)
    video.write_videofile(final_route)

    return final_route