{{- define "worldbank.labels" -}}
app.kubernetes.io/name: worldbank
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}

{{- define "worldbank.processedS3a" -}}
s3a://{{ .Values.s3.bucket }}/{{ .Values.s3.processedPrefix }}
{{- end -}}

{{- define "worldbank.modelUri" -}}
s3://{{ .Values.s3.bucket }}/{{ .Values.s3.modelPrefix }}
{{- end -}}

{{- define "worldbank.sparkSpec" -}}
type: Python
pythonVersion: "3"
mode: cluster
image: "{{ .Values.images.spark.repository }}:{{ .Values.images.spark.tag }}"
imagePullPolicy: {{ .Values.images.spark.pullPolicy }}
mainApplicationFile: "local:///opt/spark/work-dir/etl_job.py"
arguments:
  - "--indicator={{ .Values.worldbank.indicator }}"
  - "--countries={{ .Values.worldbank.countries }}"
  - "--start-year={{ .Values.worldbank.startYear }}"
  - "--end-year={{ .Values.worldbank.endYear }}"
  - "--output={{ include "worldbank.processedS3a" . }}"
  - "--state-uri=s3://{{ .Values.s3.bucket }}/{{ .Values.s3.stateKey }}"
  {{- if .Values.worldbank.incremental }}
  - "--incremental"
  {{- end }}
sparkVersion: "3.5.1"
restartPolicy:
  type: Never
driver:
  cores: {{ .Values.spark.driver.cores }}
  memory: "{{ .Values.spark.driver.memory }}"
  serviceAccount: spark-operator-spark
  env:
    - name: AWS_REGION
      value: "{{ .Values.aws.region }}"
  envSecretKeyRefs:
    AWS_ACCESS_KEY_ID:
      name: {{ .Values.aws.secretName }}
      key: AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY:
      name: {{ .Values.aws.secretName }}
      key: AWS_SECRET_ACCESS_KEY
executor:
  instances: {{ .Values.spark.executor.instances }}
  cores: {{ .Values.spark.executor.cores }}
  memory: "{{ .Values.spark.executor.memory }}"
  env:
    - name: AWS_REGION
      value: "{{ .Values.aws.region }}"
  envSecretKeyRefs:
    AWS_ACCESS_KEY_ID:
      name: {{ .Values.aws.secretName }}
      key: AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY:
      name: {{ .Values.aws.secretName }}
      key: AWS_SECRET_ACCESS_KEY
{{- end -}}
