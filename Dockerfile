FROM amazonlinux:latest

ADD SIS_diffusion_tempo_network_1_sample_timestamp.py /

RUN pip-2.7 install -r requirements.txt

CMD [ "python2", "./SIS_diffusion_tempo_network_1_sample_timestamp.py",  ]

