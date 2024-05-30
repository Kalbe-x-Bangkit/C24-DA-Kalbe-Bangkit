variable "project_id" {
  type        = string
  description = "The ID of the Google Cloud project"
  default     = "da-kalbe"
}

variable "region" {
  type        = string
  description = "The region for deploying resources"
  default     = "asia-southeast2"
}

variable "bucket_name" {
  description = "The name of the GCS bucket where the DICOM data is stored."
  default     = "da-kalbe-dicomstore"
}

variable "dataset_id" {
  type        = string
  description = "The ID of the healthcare dataset"
  default     = "dicom-dataset"
}

variable "dicom_store_id" {
  type        = string
  description = "The ID of the healthcare DICOM store"
  default     = "original-dicom"
}

variable "bq_dataset_id" {
  description = "The ID of the BigQuery dataset to create."
  default     = "bq_dicomstore"
}

variable "deid_store_id" {
  type        = string
  description = "The ID of the healthcare DICOM store for de-identified data"
  default     = "deid-original-dicom"
}

variable "deid_bq_table_id" {
  type        = string
  description = "The ID of the BigQuery table for de-identified DICOM metadata"
  default     = "deid_dicom_metadata"
}

variable "bq_table_id" {
  type        = string
  description = "The ID of the BigQuery table to create"
  default     = "dicom_metadata"
}

variable "function_name" {
  description = "The name of the Cloud Function"
  default     = "deidAndExportDicom"
}

variable "entry_point" {
  description = "The entry point for the Cloud Function"
  default     = "deidAndExportDicom"
}

variable "runtime" {
  description = "The runtime for the Cloud Function"
  default     = "nodejs18"
}

variable "ohif_viewer" {
    description = "The ohif viewer for the Cloud Run"
    default = "ohif-viewer"
}