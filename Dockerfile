FROM python:3.9
ADD Dashboard ./Dashboard/
RUN pip install dash dash-bootstrap-components pandas numpy matplotlib plotly biopython dna_features_viewer scipy quantiprot
EXPOSE 8050
CMD ["python", "./Dashboard/app.py"]
