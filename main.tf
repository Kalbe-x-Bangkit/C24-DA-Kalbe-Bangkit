# Create Healthcare Dataset
resource "google_healthcare_dataset" "dicom_dataset" {
  name     = var.dataset_id
  project  = var.project_id
  location = var.region
}

# Create Bucket
resource "google_storage_bucket" "dicom_bucket" {
  project                     = var.project_id
  name                        = var.bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
}

# Grant Permissions
resource "google_project_iam_member" "storage_object_admin" {
  project = var.project_id
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:service-2028993996@gcp-sa-healthcare.iam.gserviceaccount.com"
}

resource "google_project_iam_member" "dataset_admin" {
  project    = var.project_id
  role       = "roles/healthcare.datasetAdmin"
  member     = "serviceAccount:service-2028993996@gcp-sa-healthcare.iam.gserviceaccount.com"
}

resource "google_project_iam_member" "dicom_store_admin" {
  project    = var.project_id
  role       = "roles/healthcare.dicomStoreAdmin"
  member     = "serviceAccount:service-2028993996@gcp-sa-healthcare.iam.gserviceaccount.com"
}

resource "google_bigquery_dataset_iam_member" "bigquery_admin" {
  dataset_id = google_bigquery_dataset.bq_dataset.dataset_id
  role       = "roles/bigquery.admin"
  member     = "serviceAccount:service-2028993996@gcp-sa-healthcare.iam.gserviceaccount.com"
}

# Allow Cloud Function to publish to Pub/Sub
resource "google_project_iam_member" "function_pubsub_publisher" {
  project = var.project_id
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:service-2028993996@gcf-admin-robot.iam.gserviceaccount.com"
}

# Allow Cloud Function to access BigQuery
resource "google_project_iam_member" "function_bigquery_user" {
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:service-2028993996@gcf-admin-robot.iam.gserviceaccount.com"
}

# Allow Cloud Scheduler to publish to Pub/Sub
resource "google_project_iam_member" "scheduler_pubsub_publisher" {
  project = var.project_id
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:service-2028993996@gcp-sa-cloudscheduler.iam.gserviceaccount.com"
}

# Create DICOM Store
resource "google_healthcare_dicom_store" "dicom_store" {
  dataset = google_healthcare_dataset.dicom_dataset.id
  name    = var.dicom_store_id
}

# Create De-identified DICOM Store
resource "google_healthcare_dicom_store" "deid_dicom_store" {
  dataset = google_healthcare_dataset.dicom_dataset.id
  name    = var.deid_store_id
}

# Create BigQuery Dataset
resource "google_bigquery_dataset" "bq_dataset" {
  dataset_id                      = var.bq_dataset_id
  project                         = var.project_id
  friendly_name                   = "DICOM Metadata"
  location                        = var.region
  description                     = "Storing DICOM metadata"
  delete_contents_on_destroy      = false
}

# Create Pub/Sub Topic
resource "google_pubsub_topic" "dicom_deid_topic" {
  name = "dicom-deid-topic"
}
resource "google_storage_bucket" "bucket_function" {
  name     = "da-kalbe-cf"
  location = var.region
}

resource "google_storage_bucket_object" "archive" {
  name   = "${var.function_name}.zip"
  bucket = google_storage_bucket.bucket_function.name
  source = "../deidandexportdicom"
}

resource "google_cloudfunctions_function" "deid_and_export_dicom_function" {
  name        = var.function_name
  description = "De-identify DICOM data and export metadata to BigQuery"
  runtime     = var.runtime
  entry_point = var.entry_point

  source_archive_bucket = google_storage_bucket.bucket_function
  source_archive_object = google_storage_bucket_object.archive.name
  region                = var.region

  environment_variables = {
    DICOM_STORE_ID       = var.dicom_store_id
    DEID_STORE_ID        = var.deid_store_id
    DATASET_ID           = var.dataset_id
    BQ_DATASET_ID        = var.bq_dataset_id
    ORIGINAL_BQ_TABLE_ID = var.bq_table_id
    DEID_BQ_TABLE_ID     = var.deid_bq_table_id
    REGION               = var.region
    PROJECT_ID           = var.project_id
  }

  event_trigger {
    event_type = "google.pubsub.topic.publish"
    resource   = google_pubsub_topic.dicom_deid_topic.id
  }
}

# Create Cloud Scheduler Job
resource "google_cloud_scheduler_job" "dicom_check_job" {
  name        = "dicom-check-job"
  description = "Periodic job to check for new DICOM data and trigger Cloud Function"
  schedule    = "*/5 * * * *"  # Adjust the frequency as needed
  time_zone   = "UTC"

  pubsub_target {
    topic_name = google_pubsub_topic.dicom_deid_topic.id
    data       = base64encode("Check for new DICOM data")
  }
}

# Cloud Run OHIF Viewer
resource "google_cloud_run_service" "default" {
  name     = var.ohif_viewer
  location =  var.region

  template {
    spec {
      containers {
        image = "asia-southeast2-docker.pkg.dev/da-kalbe/ohif/viewer-google-cloud:latest"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}