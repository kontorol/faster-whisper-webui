from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip
import numpy as np
from bidi.algorithm import get_display
import arabic_reshaper

def auto_detect_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        try:
            import tensorflow as tf
            return tf.test.is_built_with_cuda()
        except ImportError:
            return False


def split_text_into_lines(data):
    MaxChars = 30
    MaxDuration = 2.5
    MaxGap = 1.5
    subtitles = []
    line = []
    line_duration = 0
    line_chars = 0
    for idx, word_data in enumerate(data):
        word = word_data["word"]
        start = word_data["start"]
        end = word_data["end"]
        line.append(word_data)
        line_duration += end - start
        temp = " ".join(item["word"] for item in line)
        new_line_chars = len(temp)
        duration_exceeded = line_duration > MaxDuration
        chars_exceeded = new_line_chars > MaxChars
        if idx > 0:
            gap = word_data['start'] - data[idx-1]['end']
            maxgap_exceeded = gap > MaxGap
        else:
            maxgap_exceeded = False
        if duration_exceeded or chars_exceeded or maxgap_exceeded:
            if line:
                subtitle_line = {
                    "word": " ".join(item["word"] for item in line),
                    "start": line[0]["start"],
                    "end": line[-1]["end"],
                    "textcontents": line
                }
                subtitles.append(subtitle_line)
            line = []
            line_duration = 0
            line_chars = 0
    if line:
        subtitle_line = {
            "word": " ".join(item["word"] for item in line),
            "start": line[0]["start"],
            "end": line[-1]["end"],
            "textcontents": line
        }
        subtitles.append(subtitle_line)
    return subtitles

def all_segments_have_words(segments):
    return all(
        isinstance(segment, dict) and
        'words' in segment and
        isinstance(segment['words'], list) and
        len(segment['words']) > 0
        for segment in segments
    )

def apply_effect(frame, effect_type, t, duration):
    if effect_type == 'fade_in':
        return frame * min(1, t / duration)
    elif effect_type == 'fade_out':
        return frame * max(0, 1 - t / duration)
    elif effect_type == 'scale':
        scale = 0.5 + 0.5 * min(1, t / duration)
        return np.repeat(np.repeat(frame, int(scale), axis=0), int(scale), axis=1)
    elif effect_type == 'rotate':
        angle = 360 * (t / duration)
        return np.rot90(frame, k=int(angle / 90))
    return frame

# GPU support can be added here if needed
USE_GPU = None  # Set this to True if you want to use GPU
USE_GPU = auto_detect_gpu() if USE_GPU is None else USE_GPU

if USE_GPU:
    try:
        import cupy as cp
        import moviepy_cuda
        from numba import jit, cuda

        moviepy_cuda.init()

        @jit(target_backend='cuda')
        def apply_effect_gpu(frame, effect_type, t, duration):
            if effect_type == 'fade_in':
                return frame * min(1, t / duration)
            elif effect_type == 'fade_out':
                return frame * max(0, 1 - t / duration)
            elif effect_type == 'scale':
                scale = 0.5 + 0.5 * min(1, t / duration)
                return cp.array(frame).repeat(int(scale), axis=0).repeat(int(scale), axis=1)
            elif effect_type == 'rotate':
                angle = 360 * (t / duration)
                return cp.rot90(cp.array(frame), k=int(angle / 90))
            return frame

        class GPUTextClip(TextClip):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.cuda_img = cp.array(self.img)

            def get_frame(self, t):
                return cp.asnumpy(self.cuda_img)

        print("GPU support enabled.")

    except ImportError:
        print("GPU libraries not found. Falling back to CPU.")
        USE_GPU = False

if not USE_GPU:
    GPUTextClip = TextClip
    apply_effect_gpu = apply_effect
    print("Using CPU for processing.")


def create_caption(textJSON, framesize, style_config=None, highlight=True):
    if style_config is None:
        style_config = {}
    
    default_style = {
        'font': "Arial",
        'fontsize': int(framesize[1] * 0.05),
        'color': 'white',
        'highlight_color': 'yellow',
        'stroke_color': 'black',
        'stroke_width': 1.5,
        'bg_color': (64, 64, 64, 128),
        'effect': None,
        'rtl': False,        
    }
    style = {**default_style, **style_config}

    rtl = style.get('rtl', False)

    # Check if custom font file is provided
    if 'font_file' in style:
        from PIL import ImageFont
        custom_font = ImageFont.truetype(style['font_file'], style['fontsize'])
        style['font'] = custom_font

    frame_width, frame_height = framesize
    x_buffer = frame_width * 0.1
    max_line_width = frame_width - 2 * x_buffer
    word_clips = []
    xy_textclips_positions = []
    x_pos, y_pos = x_buffer, frame_height * 0.8
    line_width = 0
    line_words = []

    for wordJSON in textJSON['textcontents']:
        word = wordJSON['word']
        duration = wordJSON['end'] - wordJSON['start']
        
        # Handle RTL text
        if rtl:
            word = arabic_reshaper.reshape(word)
            word = get_display(word)

        word_style = style.copy()
        if isinstance(style_config.get('effect'), dict):
            word_style['effect'] = style_config['effect'].get(word, style['effect'])

        word_clip = GPUTextClip(word, font=word_style['font'], fontsize=word_style['fontsize'],
                                color=word_style['color'], stroke_color=word_style['stroke_color'],
                                stroke_width=word_style['stroke_width'])
        word_clip = word_clip.set_start(wordJSON['start']).set_duration(duration)

        if word_style['effect']:
            word_clip = word_clip.fl(lambda gf, t: apply_effect(gf(t), word_style['effect'], t, duration))

        word_width, word_height = word_clip.size
        
        if line_width + word_width > max_line_width:
            # Arrange words in the line (RTL or LTR)
            arrange_line_words(line_words, x_buffer, y_pos, rtl, xy_textclips_positions, word_clips)
            line_words = []
            y_pos -= word_height * 1.5
            line_width = 0
        
        line_words.append({
            "clip": word_clip,
            "width": word_width,
            "height": word_height,
            "word": word,
            "start": wordJSON['start'],
            "end": wordJSON['end'],
            "duration": duration
        })
        line_width += word_width + word_style['fontsize'] * 0.2

    # Arrange the last line
    if line_words:
        arrange_line_words(line_words, x_buffer, y_pos, rtl, xy_textclips_positions, word_clips)

    if highlight:
        for highlight_word in xy_textclips_positions:
            word_clip_highlight = GPUTextClip(highlight_word['word'], font=style['font'],
                                              fontsize=style['fontsize'], color=style['highlight_color'],
                                              stroke_color=style['stroke_color'],
                                              stroke_width=style['stroke_width'])
            word_clip_highlight = word_clip_highlight.set_start(highlight_word['start']).set_duration(highlight_word['duration'])
            word_clip_highlight = word_clip_highlight.set_position((highlight_word['x_pos'], highlight_word['y_pos']))
            word_clips.append(word_clip_highlight)

    return word_clips, xy_textclips_positions

def arrange_line_words(line_words, x_buffer, y_pos, rtl, xy_textclips_positions, word_clips):
    if rtl:
        line_words.reverse()
    
    line_width = sum(word['width'] for word in line_words) + (len(line_words) - 1) * line_words[0]['clip'].w * 0.2
    x_pos = x_buffer

    for word in line_words:
        if rtl:
            x_pos = x_buffer + line_width - word['width']
        
        xy_textclips_positions.append({
            "x_pos": x_pos,
            "y_pos": y_pos,
            "width": word['width'],
            "height": word['height'],
            "word": word['word'],
            "start": word['start'],
            "end": word['end'],
            "duration": word['duration']
        })
        word['clip'] = word['clip'].set_position((x_pos, y_pos))
        word_clips.append(word['clip'])
        
        if rtl:
            x_pos -= word['width'] + word['clip'].w * 0.2
        else:
            x_pos += word['width'] + word['clip'].w * 0.2


def burn_caption(segments, srcfilename, outfilename, style_config=None):
    has_words = all_segments_have_words(segments)
    
    wordlevel_info = []
    if has_words:
        for segment in segments:
            for word in segment['words']:
                wordlevel_info.append({'word': word['word'], 'start': word['start'], 'end': word['end']})
    else:
        for segment in segments:
            wordlevel_info.append({'word': segment.get('text', ''), 'start': segment.get('start', 0), 'end': segment.get('end', 0)})

    linelevel_subtitles = split_text_into_lines(wordlevel_info)
    input_video = VideoFileClip(srcfilename)
    frame_size = input_video.size
    all_linelevel_splits = []

    for line in linelevel_subtitles:
        out_clips, positions = create_caption(line, frame_size, style_config, highlight=has_words)
        
        max_width = max(position['x_pos'] + position['width'] for position in positions) if positions else 0
        max_height = max(position['y_pos'] + position['height'] for position in positions) if positions else 0

        if style_config and 'bg_color' in style_config:
            color_clip = ColorClip(size=(int(max_width*1.1), int(max_height*1.1)),
                                   color=style_config['bg_color'])
        else:
            color_clip = ColorClip(size=(int(max_width*1.1), int(max_height*1.1)),
                                   color=(64, 64, 64, 128))
        
        color_clip = color_clip.set_opacity(.6)
        color_clip = color_clip.set_start(line['start']).set_duration(line['end']-line['start'])

        clip_to_overlay = CompositeVideoClip([color_clip] + out_clips)
        clip_to_overlay = clip_to_overlay.set_position(("center", "bottom"))
        all_linelevel_splits.append(clip_to_overlay)

    final_video = CompositeVideoClip([input_video] + all_linelevel_splits)
    final_video = final_video.set_audio(input_video.audio)
    final_video.write_videofile(outfilename, codec="libx264", audio_codec="aac", gpu=USE_GPU)
    return outfilename



# style_config = {
#         'font': "Arial",
#         'rtl': False,
#         # 'font_file': 'path/to/your/persian_font.ttf',
#         # 'fontsize': 30,
#         'color': 'white',
#         'highlight_color': 'yellow',
#         'stroke_color': 'black',
#         'stroke_width': 1.5,
#         'bg_color': (64, 64, 64, 128),
#         'effect': None,
#     }
