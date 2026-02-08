from typing import TypedDict
class HandPose(TypedDict):
    name: str
    index: int # 0. Fist 1. Point 2. Thumbs_up 3. Peace 4. Rocknroll 5. Gun 6. Open 7. Middle finger

def get_pose(json) -> HandPose:
    match json:
        case {'up_count': 0}:
            return { 'name': 'fist', 'index': 0 }
        case { 'index': True, 'up_count': 1 }:
            return { 'name': 'point', 'index': 1 }
        case { 'thumb': True, 'up_count': 1 }:
            return { 'name': 'thumbs_up', 'index': 2 }
        case { 'index': True, 'middle': True, 'up_count': 2 }:
            return { 'name': 'peace', 'index': 3 }
        case { 'index': True, 'little': True, 'up_count': count } if count in [2,3]:
            return { 'name': 'rocknroll', 'index': 4 }
        case { 'thumb': True, 'index': True, 'up_count': 2 }:
            return { 'name': 'gun', 'index': 5}
        case { 'up_count': 5 }:
            return { 'name': 'open', 'index': 6 }
        case { 'middle': True, 'up_count': 1 }:
            return { 'name': 'middle_finger', 'index': 7 }
        case _:
            return { 'name': 'unkown', 'index': -1 }