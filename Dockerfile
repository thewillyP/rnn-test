FROM thewillyp/devenv:master-1.0.23@sha256:4ad3c015a5809a914c10bb54576abfb4e200c1ae7a93dfaeb53e842921fd844e

WORKDIR /

RUN git clone https://github.com/thewillyP/rnn-test.git

WORKDIR /rnn-test

RUN chmod -R 777 /rnn-test

RUN mkdir -p /wandb_data

RUN chmod -R 777 /wandb_data

COPY entrypoint.sh entrypoint.sh

RUN chmod +x entrypoint.sh

ENTRYPOINT ["entrypoint.sh"]