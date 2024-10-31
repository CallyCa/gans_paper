FROM continuumio/miniconda3:latest

RUN apt-get update && apt-get install -y \
        protobuf-compiler \
        sudo \
        git \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml /tmp/
COPY requirements.txt /tmp/

RUN conda env create -f /tmp/environment.yml \
    && conda run -n homl3 pip install -r /tmp/requirements.txt \
    && conda clean -afy \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete \
    && rm /tmp/environment.yml \
    && rm /tmp/requirements.txt

ARG username
ARG userid

ARG home=/home/${username}
ARG workdir=${home}/handson-ml3

RUN adduser ${username} --uid ${userid} --gecos '' --disabled-password \
    && echo "${username} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${username} \
    && chmod 0440 /etc/sudoers.d/${username}

WORKDIR ${workdir}
RUN chown ${username}:${username} ${workdir}

USER ${username}
WORKDIR ${workdir}

ENV PATH /opt/conda/envs/homl3/bin:$PATH

# Enable diffing notebooks with nbdiff (and nbdiff support in git diff command)
RUN /opt/conda/envs/homl3/bin/git-nbdiffdriver config --enable --global

# Optionally ignore either metadata or details in nbdiff within git diff
# RUN git config --global diff.jupyternotebook.command '/opt/conda/envs/homl3/bin/git-nbdiffdriver diff --ignore-metadata'
RUN git config --global diff.jupyternotebook.command '/opt/conda/envs/homl3/bin/git-nbdiffdriver diff --ignore-details'

# Copy the project source code and various config directories into the container
COPY projeto-gan-autoencoder-ahp-gaussiano/src ${home}/handson-ml3/src
COPY projeto-gan-autoencoder-ahp-gaussiano/config ${home}/handson-ml3/config
COPY projeto-gan-autoencoder-ahp-gaussiano/config/updated-configs ${home}/handson-ml3/config/updated-configs
COPY projeto-gan-autoencoder-ahp-gaussiano/config/json ${home}/handson-ml3/config/json
COPY projeto-gan-autoencoder-ahp-gaussiano/config/csv ${home}/handson-ml3/config/csv
COPY projeto-gan-autoencoder-ahp-gaussiano/notebooks/images/generative ${home}/handson-ml3/images/generative
COPY projeto-gan-autoencoder-ahp-gaussiano/notebooks/save-models ${home}/handson-ml3/save-models

COPY docker/bashrc.bash /tmp/
RUN cat /tmp/bashrc.bash >> ${home}/.bashrc \
    && echo "export PATH=\"${workdir}/docker/bin:$PATH\"" >> ${home}/.bashrc \
    && sudo rm /tmp/bashrc.bash

# Optionally enable automatic save of python-only and html-only exports alongside the notebook
# COPY docker/jupyter_notebook_config.py /tmp/
# RUN cat /tmp/jupyter_notebook_config.py >> ${home}/.jupyter/jupyter_notebook_config.py
# RUN sudo rm /tmp/jupyter_notebook_config.py

# Optionally disable git diff paging
# RUN git config --global core.pager ''

# Optionally set a constant notebook URL (just localhost:8888)
# To avoid making a security hole you SHOULD regenerate the hash for your own non-empty password and replace the hash below.
# You can compute a password hash in any notebook, just run the code:
# from notebook.auth import passwd
# passwd()
# and take the hash from the output
# RUN mkdir -p ${home}/.jupyter && \
#    echo 'c.NotebookApp.password = u"sha1:c6bbcba2d04b:f969e403db876dcfbe26f47affe41909bd53392e"' \
#    >> ${home}/.jupyter/jupyter_notebook_config.py
