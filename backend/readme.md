# Projects Folders
```shell
.
├── docker-compose.yml
├── docker_command_shortcut.txt 
├── readme.md
├── DB # mysql config 파일 정의
│   ├── conf.d
│   │   └── my.cnf
│   └── mysql
├── RProxy # 리버스 프록시 config 파일 정의
│   └── nginx.conf
├── diaryServer # 다이어리 서버 디렉토리
│   ├── Dockerfile
│   ├── code
│   │   └── app.py
│   └── command.sh
├── modelServer # 모델 서버 디렉토리
│   ├── DockerFile
│   └── code
│       └── app.py
├── tables # 각 서버를 위한 공통 코드, orm 객체, session 반환함수, 예외처리 데코레이터가 정의됨
│   ├── session.py
│   ├── tables.py # orm 객체 정의
│   └── utils.py
└── template # 서버 공통으로 사용할 Dockerfile 및 사용 패키지 정의 폴더
    ├── Dockerfile
    └── requirements.txt

```

# Model Arch
<img width="2342" height="1157" alt="Image" src="https://github.com/user-attachments/assets/27afddbf-10b2-4477-9ea3-46324c0a579c" />

