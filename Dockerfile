FROM thewillyp/devenv:master-1.0.28-gpu@sha256:12d8d4245f725d0576dc178df4fa4bbfa44a8e6cb3da9752feb9ae8590c24b4e

WORKDIR /

RUN git clone https://github.com/thewillyP/rnn-test.git

WORKDIR /rnn-test

RUN chmod -R 777 /rnn-test

RUN mkdir -p /wandb_data

RUN chmod -R 777 /wandb_data

COPY entrypoint.sh entrypoint.sh

RUN chmod +x entrypoint.sh

ENTRYPOINT ["entrypoint.sh"]