# Custom Grafana image with RAG monitoring provisioned
# Build: docker build -f grafana/Dockerfile -t rag-grafana .
FROM grafana/grafana:11.2.0

COPY provisioning/datasources/pg-datasource.yml /etc/grafana/provisioning/datasources/
COPY provisioning/dashboards/dashboards.yml /etc/grafana/provisioning/dashboards/
COPY dashboards/rag-monitoring.json /etc/grafana/dashboards/
