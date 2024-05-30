terraform {
  required_version = ">= 0.13"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 4.76.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = ">= 4.76.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
