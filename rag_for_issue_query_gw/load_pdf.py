#from google.cloud import storage

project = "xxxxx"
bucket = "bucket"

#creds = service_account.Credentials.from_service_account_file(key_file)
#storage_client = storage.Client(credentials = creds, project = project)
#bucket_gcs = storage_client.get_bucket(bucket)
#from pypdf import PdfReader

def pdf_read(file):
    import gcsfs
    from pypdf import PdfReader

    gcs_file_system = gcsfs.GCSFileSystem(project=project)
    gcs_pdf_path = f"gs://{bucket}/{file}"
    print(gcs_pdf_path)

    f_object = gcs_file_system.open(gcs_pdf_path, "rb")
    txt = ""
    reader = PdfReader(f_object)
    pages = reader.pages
    for i in range(len(pages)):
        txt += pages[i].extract_text()
        
    return txt


