import os
import io
from typing import List, Dict, Any, Tuple, Optional
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from src.connectors.base import BaseConnector
from src.core.logger import setup_logger

logger = setup_logger(__name__)

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

class GoogleDriveConnector(BaseConnector):
    def __init__(self, credentials_path: str = "credentials.json", token_path: str = "token.json"):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.creds = None
        self.service = None

    def authenticate(self) -> Any:
        """Authenticate using local credentials.json and cache token.json."""
        if os.path.exists(self.token_path):
            try:
                self.creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)
            except Exception as e:
                logger.warning(f"Failed to load cached credentials: {e}")
        
        # If there are no (valid) credentials available, let the user log in.
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                try:
                    self.creds.refresh(Request())
                except Exception as e:
                    logger.warning(f"Failed to refresh credentials: {e}")
                    self.creds = None
            
            if not self.creds:
                if not os.path.exists(self.credentials_path):
                    raise FileNotFoundError(
                        f"Google OAuth credentials file not found at: {self.credentials_path}. "
                        "Please download the credentials.json file from the Google Cloud Console "
                        "and place it in the project root."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
                self.creds = flow.run_local_server(port=0)
            
            # Save the credentials for the next run
            try:
                with open(self.token_path, 'w') as token:
                    token.write(self.creds.to_json())
            except Exception as e:
                logger.error(f"Failed to save cached credentials to {self.token_path}: {e}")
        
        self.service = build('drive', 'v3', credentials=self.creds)
        return self.service

    def list_files(self, folder_id: str = None) -> List[Dict[str, Any]]:
        """List files and folders in Google Drive, optionally inside a specific folder."""
        if not self.service:
            self.authenticate()
            
        # Select active files and filter by parent directory (default to root)
        query = "trashed = false"
        if folder_id:
            query += f" and '{folder_id}' in parents"
        else:
            query += " and 'root' in parents"
            
        results = self.service.files().list(
            q=query,
            pageSize=100,
            fields="nextPageToken, files(id, name, mimeType, modifiedTime, version, size)"
        ).execute()
        
        return results.get('files', [])

    def download_file(self, file_id: str, mime_type: str) -> Tuple[bytes, str]:
        """
        Download a file by ID. Handles Google Workspace format conversions.
        Returns: (bytes, extension)
        """
        if not self.service:
            self.authenticate()
            
        # Google Workspace MIME types mapping to export formats
        export_mapping = {
            "application/vnd.google-apps.document": (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ".docx"
            ),
            "application/vnd.google-apps.spreadsheet": (
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ".xlsx"
            ),
            "application/vnd.google-apps.presentation": (
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                ".pptx"
            )
        }
        
        if mime_type in export_mapping:
            target_mime, ext = export_mapping[mime_type]
            request = self.service.files().export_media(fileId=file_id, mimeType=target_mime)
        else:
            # Fetch the actual filename to get its extension
            meta = self.service.files().get(fileId=file_id, fields="name").execute()
            name = meta.get("name", "")
            ext = os.path.splitext(name)[1].lower() if "." in name else ".txt"
            request = self.service.files().get_media(fileId=file_id)
            
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            
        return fh.getvalue(), ext

    def fetch_documents(self, selected_files: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Download and package files for ingestion.
        Returns a list of dicts: {"raw_data": bytes, "source": str, "extension": str, "metadata": dict}
        """
        if not self.service:
            self.authenticate()
            
        docs = []
        for file in (selected_files or []):
            file_id = file["id"]
            name = file["name"]
            mime_type = file["mimeType"]
            modified_time = file.get("modifiedTime")
            version = file.get("version")
            
            try:
                raw_data, ext = self.download_file(file_id, mime_type)
                
                # Clean source name to include extension
                source_name = name + ext if not name.endswith(ext) else name
                
                # Construct standard metadata dictionary
                meta = {
                    "source_type": "google_drive",
                    "drive_file_id": file_id,
                    "version": str(version) if version is not None else "1",
                    "modified_time": modified_time,
                    "source": source_name
                }
                
                docs.append({
                    "raw_data": raw_data,
                    "source": source_name,
                    "extension": ext,
                    "metadata": meta
                })
            except Exception as e:
                logger.error(f"Failed to download Google Drive file {name} ({file_id}): {e}")
                
        return docs
