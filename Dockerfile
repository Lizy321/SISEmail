FROM amazonlinux:latest

ADD SIS_diffusion_tempo_network_1_sample_timestamp.py /

RUN yum -y install which unzip aws-cli

RUN curl -s https://bootstrap.pypa.io/get-pip.py | python

RUN yum install -y \
  gcc \
  gcc-gfortran \
  lapack-devel \
  gcc-c++ \
  findutils \
  python27-devel

COPY requirements.txt .

RUN pip install -r requirements.txt

CMD [ "python", "./SIS_diffusion_tempo_network_1_sample_timestamp.py", "--x", "50", "--ratio", "0.75", "--email", "1", "--beta", "1", "--gamma", "0.01", "--timestep", "100" ]

