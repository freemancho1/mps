import copy
import difflib
from typing import Any


# 한 줄에 표시할 '사용 가능한 속성' 최대 개수 (그 이상은 ... 로 생략)
DD_MAX_KEYS_SHOWN = 20


def _missing_key_message(owner: str, key: str, keys: list, hint: str) -> str:
    """ 누락 키 접근 시: '어느 객체에서 / 무엇이 없고 / 무엇을 쓸 수 있는지'를 모두 노출.

    msg.bt.err.enrty_reject 처럼 중첩 점 표기에서 오타가 나도 전체 경로·후보·추천을
    한 번에 보여줘 어느 단계가 틀렸는지 즉시 알 수 있게 한다.
    """
    shown = keys[:DD_MAX_KEYS_SHOWN]
    avail = ', '.join(f"'{k}'" for k in shown)
    if len(keys) > DD_MAX_KEYS_SHOWN:
        avail += f", ... (총 {len(keys)}개)"
    msg = f"{owner} '{key}' 속성이 존재하지 않습니다."
    if hint:
        msg += f" (혹시 '{hint}' 입니까?)"
    msg += f"\n    → 사용 가능한 속성: [{avail}]" if keys else "\n    → 사용 가능한 속성이 없습니다(빈 객체)."
    return msg


DD_MSG = {
    'KEY_DOES_NOT_EXIST'            : _missing_key_message,
    'INTERNAL_PROPERTY_READ_ONLY'   : lambda key: f'내부 속성 \'{key}\'는 값을 변경할 수 없습니다.',
    'ERROR'                         : lambda err: f'딕셔너리를 점 표기법으로 변환하는 과정에서 오류가 발생했습니다.\n[ERROR]: {err}',
}
DD_DEFAULT = {
    'ENSURE_ASCII'  : False,
    'PRINT_INDENT'  : 8,
    'PRINT_TYPE'    : 'json'   # json or not (dict)
}

class DictDot(dict):    
    """ 딕셔너리 데이터를 '점 표기법(Dot Notation)'으로 변환하는 클래스
        
        주요 기능:
            - 점 표기법 변환: 재귀적 방법으로 순환 처리
            - 딕셔너리로 재 변환: 변경된 점 표기법 객체를 딕셔너리로 재 전환
            - json 변환: 변경된 점 표기법 객체를 딕셔너리로 전환 후 json으로 전환
    """
    
    def __init__(self, *args, **kwargs):
        # 내부에서만 사용하는 속성으로 __setattr__을 우회(상위 __setattr__ 사용)하여 설정
        _name_ = kwargs.pop('_name_', '')
        object.__setattr__(self, '_name_', _name_)
        object.__setattr__(self, '_sep_', '.')
        
        # *args(거의 안씁), **kwargs를 이용해 딕셔너리 생성
        super().__init__(*args, **kwargs)
        
        for key, value in self.items():
            if isinstance(value, dict):
                # __setattr__()이 아닌 _setitem__() 함수에서 수행(로직 통일)
                self[key] = value 
                
    def __getattr__(self, key):
        # __deepcopy__·_ipython_* 등 파이썬/IDE 내부 프로브는 조용히 표준 AttributeError로 처리.
        # (메시지 키는 밑줄로 시작하지 않으므로 사용자 키 탐색에는 영향 없음)
        if key.startswith('_'):
            raise AttributeError(key)
        try:
            return self[key]
        except KeyError:
            owner = self._owner_desc()
            keys = sorted(self.keys())
            # difflib로 유사 키 추천 (enrty_reject → entry_reject)
            matches = difflib.get_close_matches(key, [str(k) for k in keys], n=1, cutoff=0.6)
            hint = matches[0] if matches else ''
            raise AttributeError(
                DD_MSG['KEY_DOES_NOT_EXIST'](owner, key, keys, hint)
            ) from None
        except Exception as err:
            raise RuntimeError(DD_MSG['ERROR'](err))
        
    def __setattr__(self, key, value):
        # __setitem__()으로 로직 통일
        self[key] = value 
        
    def __setitem__(self, key, value):
        if key in ('_name_', '_sep_'):
            raise RuntimeError(DD_MSG['INTERNAL_PROPERTY_READ_ONLY'](key))
        
        if isinstance(value, dict):
            child_name = f'{self._name_}{self._sep_}{key}' if self._name_ else key
            value = DictDot(value, _name_=child_name)
        
        # 실제 변경은 dict 클래스에 위임
        super().__setitem__(key, value)
        
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            owner = self._owner_desc()
            keys = sorted(self.keys())
            matches = difflib.get_close_matches(key, [str(k) for k in keys], n=1, cutoff=0.6)
            raise AttributeError(
                DD_MSG['KEY_DOES_NOT_EXIST'](owner, key, keys, matches[0] if matches else '')
            ) from None
        except Exception as err:
            raise RuntimeError(DD_MSG['ERROR'](err))
        
    def __repr__(self):
        return f'DictDot({super().__repr__()})'
    
    def _get_name(self):
        return f'\'{self._name_}\' ' if self._name_ else ''
    
    def _owner_desc(self):
        """ 누락 키 메시지용 소유 객체 설명. 알고 있는 점 경로(_name_)를 그대로 노출.

        중첩 자식은 부모가 이름을 가질 때 'bt.err'처럼 점 경로가 누적된다.
        최상위(변수에 직접 대입된) 객체는 이름을 알 수 없어 '최상위(root)'로 표기하되,
        '사용 가능한 속성' 목록이 어느 단계인지 식별을 보완한다.
        """
        return f"'{self._name_}' 객체에" if self._name_ else "최상위(root) 객체에"
    
    def copy(self):
        """ 얕은 복사를 수행(주소값만 복사) """
        return self.__class__(self)
    
    def deepcopy(self):
        """ 깊은 복사를 수행(내부의 모든 객체까지 재귀적으로 복사해 독립된 복사본 리턴) """
        return self.__class__(copy.deepcopy(dict(self)))
    
    def to_dict(self):
        result = {}
        for key, value in self.items():
            result[key] = value.to_dict() if isinstance(value, DictDot) else value
        return result
    
    def print(self, **kwargs):
        """ 객체를 JSON 형태로 출력함 """
        type = kwargs.get('type', DD_DEFAULT['PRINT_TYPE'])
        indent = kwargs.get('indent', DD_DEFAULT['PRINT_INDENT'])
        ensure_ascii = kwargs.get('ensure_ascii', DD_DEFAULT['ENSURE_ASCII'])
        
        def _print(node: Any, depth: int = 0):
            tab_size = depth * indent
            for key, value in node.items():
                print(f'{" "*tab_size}{key}: ', end='')
                if isinstance(value, dict):
                    print()
                    depth += 1
                    _print(value, depth)
                    depth -= 1
                    if depth == 0:
                        print()
                else:
                    print(f'\'{value}\'' if isinstance(value, str) else value)
        
        _print(self.to_dict())
        # if type == DD_DEFAULT['PRINT_TYPE']:
        #     print(json.dumps(self.to_dict(), indent=indent, ensure_ascii=ensure_ascii))
        # else:
        #     print(self.to_dict())
