FROM thewillyp/devenv:master-1.0.19@sha256:fb9d0830a7239a4e977eea92ff886c9d5ca818f5148b5103ef50bf730248bdf2

WORKDIR /run

COPY ./src ./

COPY .git/ ./.git/