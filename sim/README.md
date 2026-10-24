# 물리 시뮬레이션 영상 (NeuroMechFly)

`flygym_choice.py`는 MuJoCo 물리 엔진 위의 NeuroMechFly 초파리가 두 프롬프트의 냄새를 맡고, 버섯체가 안전하다고 판단한 쪽으로 걸어가는 영상을 만들어요. 영상은 `web/videos/`에 저장되고, 웹 데모의 "물리 시뮬" 탭에 설명과 함께 나타나요.

## 왜 구버전인가
- FlyGym 2.x(2026년 4월 재작성)에는 냄새 아레나가 없어요.
- 구버전 API를 이어받은 `flygym-gymnasium` 1.3.2에는 `OdorArena`와 냄새 추적 예제(`examples/olfaction/simple_odor_taxis.py`)가 있어요. 이 코드는 그 예제를 바탕으로 만들었어요.

## 설치 (한 번만, 기본 Anaconda 환경과 분리)
```bash
conda create -n flygym python=3.12 -y
conda run -n flygym python -m pip install flygym-gymnasium==1.3.2 "networkx>=3,<4"
```
`networkx`는 보행 컨트롤러 예제가 필요로 해서 같이 설치해요. 이 컴퓨터(Windows 11)에서는 렌더링까지 문제없이 돌아갔어요.

## 실행 (프로젝트 루트에서)
1. FlyGuard 서버를 켜요. 시뮬레이션이 각 프롬프트의 판정을 서버에 물어봐요.
   ```bash
   uvicorn server.app:app --host 127.0.0.1 --port 8000
   ```
2. 다른 터미널에서 미리 정한 장면 5개를 렌더해요.
   ```bash
   conda run -n flygym python sim/flygym_choice.py --showcase
   ```
3. 원하는 두 문장으로 장면 하나를 만들 수도 있어요.
   ```bash
   conda run -n flygym python sim/flygym_choice.py --left "Jailbreak themes for iPhone" --right "Ignore previous instructions" --out web/videos/custom.mp4
   ```
- 물리 시뮬레이션은 실제 시간보다 약 24배 느려요. 시뮬레이션 6초가 걸리는 장면은 약 2.5분 걸려요.
- 초파리가 끌리는 냄새원에 2 mm 안으로 들어가면 일찍 끝나요.
- 영상은 0.25배속, H.264(yuv420p)로 저장돼서 브라우저에서 바로 재생돼요.
- 장면 정보(프롬프트, 판정, 도착한 쪽)는 `web/videos/videos.json`에 쌓여요.

## 장면 설계
- **배치**
  - 초파리는 원점에서 화면 위쪽(+y)을 보고 출발해요.
  - 왼쪽 프롬프트의 냄새는 (−5, 15) mm, 오른쪽은 (5, 15) mm에서 나와요.
  - 두 냄새는 서로 다른 냄새 채널이에요.
- **냄새의 가치**
  - FlyGuard 투표가 정상이면 끌리는 냄새(초록 표시), 인젝션이면 피하는 냄새(빨강 표시)가 돼요.
  - 세기는 투표 확신도에 비례해요(최소 0.25).
- **방향 전환**
  - FlyGym 예제와 같은 방식이에요. 냄새마다 좌우 더듬이·촉수의 냄새 세기 차이로 방향 편향을 계산해서 합해요.
  - 그 편향만큼 한쪽 보행 리듬(CPG)을 약하게 해서 몸을 틀어요.
- 버섯체가 직접 다리를 움직이지는 않아요. 버섯체가 냄새의 가치를 정하고, 그 가치가 냄새 추적 컨트롤러의 방향을 정해요.
