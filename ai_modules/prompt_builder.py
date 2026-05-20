"""
Video prompt engineering for Kling i2v.

Converts static scene narrative data into motion-focused prompts that tell
Kling exactly what the character is doing, how the camera moves, and what
the environment looks like in motion.

Structure: [character + primary action] [camera] [environment animation] [mood] [style]
"""

import re
from typing import List, Optional

# ---------------------------------------------------------------------------
# Action verb → specific character body motion
# ---------------------------------------------------------------------------
_VERB_MOTION = {
    'walk':     'walking forward with natural arm swing and easy relaxed stride',
    'run':      'running with arms pumping, feet lifting off the ground, leaning forward with speed',
    'sit':      'bending knees and lowering body into a sitting position, settling comfortably',
    'stand':    'standing upright, weight shifting slightly from foot to foot, breathing naturally',
    'look':     'turning head slowly, eyes scanning the scene with focused curiosity',
    'notice':   'stopping mid-step, head snapping to the side, eyes widening in sudden discovery',
    'see':      'eyes focusing, head tilting toward the subject of attention, brows slightly raised',
    'clean':    'bending at the waist and reaching both hands out, scrubbing and lifting debris',
    'pick':     'bending knees, one hand reaching down and fingers closing around an object, lifting it',
    'smile':    'cheeks lifting, eyes crinkling at the corners, warm gentle smile spreading across face',
    'laugh':    'shoulders bouncing, head tilting back slightly, mouth open with joy',
    'cry':      'head drooping forward, shoulders trembling, one hand rising to wipe a tear from cheek',
    'help':     'stepping forward, one arm extending outward with an open welcoming gesture',
    'climb':    'arms reaching upward and gripping, legs pushing off as body pulls itself up step by step',
    'open':     'hand reaching forward and fingers gripping an edge, pushing or pulling it open',
    'find':     'eyes widening with surprise and delight, body leaning forward, hand reaching out',
    'jump':     'knees bending deep then legs launching upward, arms raised for momentum',
    'wave':     'arm rising smoothly, hand opening and moving back and forth in a cheerful greeting',
    'hug':      'arms spreading wide then wrapping around, body pulling close in an embrace',
    'play':     'bouncing lightly on feet, arms gesturing freely, face animated with spontaneous energy',
    'work':     'hands moving with focused purpose, body leaning into the task with concentration',
    'talk':     'mouth moving expressively, hands gesturing with emphasis while speaking',
    'build':    'hands carefully picking up pieces and placing them into position one by one',
    'carry':    'arms holding a load against chest, adjusting balance while walking carefully',
    'fall':     'body stumbling, arms shooting out to catch balance, weight shifting suddenly',
    'throw':    'arm drawing back behind shoulder, torso rotating, then arm launching forward in a full arc',
    'catch':    'arms reaching wide, eyes tracking the object, hands closing the moment it arrives',
    'eat':      'hand moving deliberately toward mouth, jaw chewing slowly and contentedly',
    'sleep':    'body relaxing completely, eyes closing softly, chest rising and falling in slow breathing',
    'dance':    'body swaying with a natural rhythm, arms flowing outward, feet stepping lightly',
    'read':     'eyes moving steadily across a page, finger tracing along the lines of text',
    'write':    'hand moving in careful controlled strokes, head bowed in concentration',
    'plant':    'kneeling with knees on the earth, hands pressing seeds gently down into the soil',
    'watering': 'tilting a container, a graceful arc of water streaming down onto the ground',
    'harvest':  'hands gripping produce firmly and pulling it free from the stem with a smooth twist',
    'cook':     'hand stirring a pot in slow circles, leaning over with a focused attentive expression',
    'draw':     'hand moving in light deliberate strokes across a surface, eyes flicking up for reference',
    'sing':     'chest expanding, mouth opening wide, head lifting slightly as the voice carries',
    'listen':   'head tilting to one side, eyes closing partway, whole body going still and attentive',
    'think':    'head tilting back, one hand raised to chin, eyes drifting upward in quiet reflection',
    'decide':   'posture straightening suddenly, chin lifting, eyes brightening with clear resolve',
    'discover': 'leaning forward with eyes wide, mouth forming a silent "oh" of pure wonder',
    'push':     'both hands pressing firmly against a surface, feet planting wide for leverage',
    'pull':     'hands gripping something, body leaning back, arms and legs working together with effort',
    'pour':     'arm tilting a vessel slowly, liquid streaming in a smooth arc downward',
    'give':     'arm extending fully forward, hand open and flat, offering something with care',
    'receive':  'both hands cupping forward together, eyes warm, accepting what is offered',
    'protect':  'stepping forward and spreading arms wide, standing firmly in front of others',
    'search':   'head turning left then right, one hand raised to shade eyes, scanning the horizon',
    'wonder':   'head tilting back, eyes wide and sparkling, mouth parting in quiet amazement',
    'cheer':    'arms punching upward toward the sky, body bouncing with irrepressible celebration',
    'share':    'holding something out in both hands, turning toward another with a generous smile',
    'learn':    'leaning forward attentively, eyes wide, nodding slowly as understanding arrives',
    'explore':  'moving forward cautiously, eyes wide, head turning to take everything in',
    'hide':     'crouching low, body curling inward, eyes peering carefully around the edge',
    'show':     'pointing toward something with one arm extended, head turning to check the reaction',
    'fix':      'bending over the object, both hands working carefully with precise small movements',
    'collect':  'bending repeatedly, picking up items and placing them into a bag or pile',
    'follow':   'moving forward with eyes locked on the figure ahead, matching their pace',
    'lead':     'striding forward confidently, one arm reaching back in invitation for others to follow',
    'save':     'lunging forward urgently, arms reaching out with determined speed',
    'celebrate': 'jumping and spinning with arms flung wide, face lit with pure happiness',
    'lift':     'bending knees, gripping firmly with both hands, then straightening up with effort to raise the object',
    'raise':    'arms pressing upward, lifting something with deliberate careful force',
    'free':     'hands working quickly to loosen and release, stepping back as the thing is let go',
    'release':  'hands opening and drawing back, letting something go gently',
    'escape':   'spinning and stepping quickly away, moving with urgency and relief',
    'flee':     'turning and running fast, legs pumping, glancing back over one shoulder',
    'shake':    'head moving quickly from side to side, or hands trembling slightly with emotion',
    'grab':     'hand shooting out and fingers closing tightly around the object',
    'reach':    'arm extending fully outward, fingers stretching as far as they can go',
    'struggle': 'body straining with effort, arms and legs working hard against resistance',
    'rescue':   'moving decisively toward someone, arms reaching out to pull them to safety',
    'escape':   'stepping backward quickly, spinning and moving away with relieved urgency',
    'return':   'walking forward with purpose, a familiar landscape coming into view',
    'arrive':   'stepping forward and slowing to a stop, looking around and taking it all in',
    'stand':    'standing upright, weight shifting slightly from foot to foot, breathing naturally',
    'pause':    'feet stopping, body going still, head tilting in a moment of reflection',
    'look back':'head turning back over the shoulder, eyes softening with memory or longing',
    'kneel':    'legs bending and lowering until knees touch the ground, hands resting forward',
    'bow':      'upper body bending forward respectfully, head lowering slowly',
    'point':    'one arm extending with a single finger directed firmly toward something',
    'nod':      'head moving slowly downward and back up in a gesture of understanding',
    'stare':    'body going still, eyes fixed wide and unblinking on something ahead',
}

# ---------------------------------------------------------------------------
# Setting keyword → what moves in that environment
# ---------------------------------------------------------------------------
_ENV_ANIMATION = {
    'farm':     'chickens bob and peck at the ground, cows flick their tails and chew, grass sways gently',
    'field':    'tall grass ripples in long rolling waves, wildflowers nod their heads in the breeze',
    'well':     'water ripples inside the well, small drops fall from the mossy stone rim',
    'forest':   'leaves tremble and rustle overhead, dappled sunlight shifts and moves through branches',
    'tree':     'branches sway in gentle arcs, individual leaves flutter and catch the light',
    'river':    'water flows and sparkles over smooth stones, reeds lean gracefully with the current',
    'stream':   'water babbles and tumbles over pebbles, small ripples fan outward in rings',
    'ocean':    'waves roll in with steady rhythm, white foam spreads across the wet sand',
    'beach':    'gentle waves lap the shoreline repeatedly, seabirds wheel and glide overhead',
    'garden':   'flowers sway on their stems, a butterfly drifts lazily past on the warm breeze',
    'mountain': 'clouds drift slowly across distant peaks, wind bends the highland grass in sweeping waves',
    'school':   'dust motes float and spin in shafts of golden light from the windows',
    'house':    'curtains billow softly inward from an open window, shadows shift slowly on the walls',
    'home':     'warm lamplight flickers gently, cosy shadows dance across the familiar walls',
    'market':   'colorful fabrics ripple in the breeze, distant figures move through the background',
    'night':    'stars twinkle and slowly wheel across the sky, fireflies drift and pulse with soft light',
    'rain':     'raindrops streak downward in curtains, puddles bloom with expanding rings on impact',
    'snow':     'snowflakes drift and spin lazily, settling softly on every surface',
    'desert':   'sand grains skitter across the dunes, heat shimmer rises in wavering columns',
    'sky':      'clouds drift and slowly reshape, birds wheel in long graceful arcs',
    'road':     'dust rises and settles behind each footstep, small pebbles scatter to the sides',
    'bridge':   'water flows and shimmers beneath, reflections ripple and break in the current',
    'barn':     'motes of hay dust float in shafts of golden afternoon light',
    'village':  'chimney smoke curls gently upward, chickens wander freely in the soft background',
    'path':     'leaves and small stones shift underfoot, light filters through overhanging branches',
    'cave':     'torchlight or sunlight flickers on the rock walls, droplets fall in slow rhythm',
    'castle':   'banners ripple in the wind, birds circle the tall towers in the distance',
    'meadow':   'long grass undulates in the breeze, small insects drift above the flower tops',
    'pond':     'water shimmers with reflected light, lily pads rock gently on small ripples',
    'bakery':   'steam curls from freshly baked goods, warm golden light fills the cozy space',
    'library':  'dust motes drift in beams of light, book pages flutter faintly near an open window',
    'workshop': 'small wood or metal shavings drift through shafts of light, tools gleam',
    'dock':     'boats rock gently on the water, ropes creak, seagulls circle overhead',
    'rooftop':  'wind ruffles hair and clothes, the city or landscape stretches out below',
}

# ---------------------------------------------------------------------------
# Camera movement — driven by verbs and scene position
# ---------------------------------------------------------------------------
_DYNAMIC_VERBS   = {'run', 'jump', 'fall', 'throw', 'catch', 'dance', 'chase', 'save', 'celebrate', 'cheer'}
_TRAVEL_VERBS    = {'walk', 'carry', 'climb', 'push', 'pull', 'explore', 'follow', 'lead', 'search', 'collect'}
_DISCOVER_VERBS  = {'notice', 'find', 'discover', 'wonder', 'see', 'learn', 'show'}
_EMOTIONAL_VERBS = {'smile', 'laugh', 'hug', 'wave', 'sing', 'celebrate', 'cheer', 'cry', 'share'}
_QUIET_VERBS     = {'sleep', 'listen', 'think', 'read', 'sit', 'look'}
_ACTION_VERBS    = {'work', 'build', 'clean', 'cook', 'plant', 'harvest', 'write', 'draw', 'fix', 'hide'}
_SOCIAL_VERBS    = {'talk', 'give', 'receive', 'help', 'protect', 'decide'}


def _camera_move(verbs: List[str], position: int, total: int) -> str:
    verb_set = set(verbs)

    if position == 1:
        return 'wide establishing shot slowly pushing in toward the character as the scene opens'
    if position == total:
        return 'camera gently pulls back to reveal the full environment, lingering warmly as the story concludes'

    if verb_set & _DYNAMIC_VERBS:
        return 'dynamic tracking camera follows the character with energy, slight motion blur on fast moments'
    if verb_set & _TRAVEL_VERBS:
        return 'camera follows alongside at a comfortable medium distance with a smooth gentle side-scroll'
    if verb_set & _DISCOVER_VERBS:
        return 'slow deliberate push in toward the character\'s face, capturing the exact moment of discovery'
    if verb_set & _EMOTIONAL_VERBS:
        return 'warm slow zoom in, camera holds softly on the character\'s expressive face'
    if verb_set & _QUIET_VERBS:
        return 'gentle slow push in, intimate and unhurried, letting the quiet emotion breathe'
    if verb_set & _ACTION_VERBS:
        return 'medium shot showing full-body action with a subtle handheld feel that grounds the movement'
    if verb_set & _SOCIAL_VERBS:
        return 'medium two-shot framing with a slow gentle zoom to draw attention to the connection between characters'

    return 'smooth gentle pan across the scene, slow and cinematic, letting details reveal themselves'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_verbs(text: str) -> List[str]:
    text_lower = text.lower()
    found = []
    for verb in _VERB_MOTION:
        if re.search(r'\b' + verb + r'(?:ed|ing|s|es)?\b', text_lower):
            found.append(verb)
    return found


def _extract_env(text: str) -> str:
    text_lower = text.lower()
    for keyword, animation in _ENV_ANIMATION.items():
        if keyword in text_lower:
            return animation
    return 'a gentle breeze moves through the air, soft ambient light shifts naturally'


def _condense_character(anchor: str) -> str:
    """Strip style/quality descriptors, keep only visual appearance."""
    if not anchor:
        return 'the main character'

    strip_phrases = [
        'chibi proportions', 'large anime eyes', 'flat cel-shaded',
        'thick black outlines', 'cartoon illustration', "children's picture book style",
        'children\'s picture book', 'highly detailed', 'sharp crisp outlines',
        'vibrant saturated colors', 'professional', 'consistent appearance',
        'secondary characters visible when present', 'same character consistent appearance throughout',
        'main character centered in frame', 'full body head to toe visible',
        'child-sized compared to surroundings', 'the described child is the main character',
        'occupying 60% of frame height',
    ]
    result = anchor
    for phrase in strip_phrases:
        result = re.sub(re.escape(phrase), '', result, flags=re.IGNORECASE)
    result = re.sub(r',\s*,+', ',', result)
    result = re.sub(r'\s+', ' ', result).strip().strip(',').strip()
    if len(result) > 160:
        result = result[:160].rsplit(',', 1)[0].strip()
    return result or 'the main character'


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_video_prompt(
    text: str,
    character_anchor: str,
    position: int,
    total_scenes: int,
    title: str = '',
) -> str:
    """
    Build a motion-focused Kling i2v prompt from scene narrative data.

    Args:
        text:             Scene narration text (what happens in the scene).
        character_anchor: Character appearance description from story JSON.
        position:         1-based position of this scene in the story.
        total_scenes:     Total number of scenes in the story.
        title:            Story title (used for mood context).

    Returns:
        A prompt string structured for Kling v2.1 image-to-video generation.
    """
    verbs    = _extract_verbs(text)
    env_anim = _extract_env(text)
    camera   = _camera_move(verbs, position, total_scenes)
    char     = _condense_character(character_anchor)

    # Character + action line
    if verbs:
        primary_motion = _VERB_MOTION[verbs[0]]
        if len(verbs) > 1:
            secondary_motion = _VERB_MOTION[verbs[1]]
            action_line = f'{char}, {primary_motion}, then {secondary_motion}.'
        else:
            action_line = f'{char}, {primary_motion}.'
    else:
        # No known verb — use the first sentence of the text as a paraphrase
        first_sentence = text.split('.')[0].strip()
        if first_sentence:
            action_line = f'{char}: {first_sentence[:120]}.'
        else:
            action_line = f'{char} in a children\'s story scene.'

    # Mood by position
    if position == 1:
        mood = 'warm and inviting opening mood, full of curiosity and gentle wonder'
    elif position == total_scenes:
        mood = 'heartwarming and satisfying conclusion, a quiet sense of accomplishment and joy'
    elif position == total_scenes // 2:
        mood = 'turning point atmosphere, energy building toward resolution'
    else:
        mood = 'engaging story-driven mood, character fully immersed in the moment'

    prompt = (
        f'{action_line} '
        f'{camera.capitalize()}. '
        f'{env_anim.capitalize()}. '
        f'{mood.capitalize()}. '
        f'Disney-Pixar 3D CGI animation, the character is a rendered 3D figure with '
        f'volume and weight, fluid natural motion, soft volumetric lighting, '
        f'cinematic depth of field. Character stays centered, full body in frame.'
    )

    return prompt
