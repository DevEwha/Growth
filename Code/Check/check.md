```markdown
# Conda 환경 재생성

## 환경 생성
```
conda env create -f /home/devewha/Growth/Growth/Code/Check/environment.yml
```

## 환경 활성화
```
conda activate <환경이름>
```

## 기존 환경이 있을 때
```
# 기존 환경 삭제
conda env remove -n <환경이름>

# 다시 생성
conda env create -f /home/devewha/Growth/Growth/Code/Check/environment.yml
```
```