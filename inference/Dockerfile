FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime 

ENV prefetch_factor=None

RUN apt update; exit 0
RUN apt install -y nano
RUN apt-get install -y  libgl1 libglib2.0-0 git  

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt install git-lfs && git-lfs install
RUN apt install -y zip
# create app directory and user
RUN mkdir /app
RUN chmod a+rw /app

#replace 1001/1007 with your user/group id 
# get uid: id -u
# get gid: id -g 
# RUN groupadd -g 1001 app && useradd -r -u 1007 -g app -d /app -s /sbin/nologin -c "non-root app user" app


WORKDIR /workspace
# RUN chown -R app:app /workspace
# RUN chmod -R a+rw /workspace


# COPY --chown=app:app ../requirements.txt /workspace
# COPY --chown=app:app ../src-python /workspace
# COPY --chown=app:app app.py /workspace

COPY requirements.txt /workspace

RUN python -m pip install --upgrade pip && pip install --upgrade --root-user-action=ignore -r requirements.txt

COPY app.py /workspace
# COPY models /workspace/models

