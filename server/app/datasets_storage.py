import os
import pandas as pd
from io import BytesIO
from fastapi import UploadFile


class DatasetStorage:
    def __init__(self, storage_folder):
        """
        Initialize the DatasetStorage class.
        """
        self.storage_folder = storage_folder
        os.makedirs(self.storage_folder, exist_ok=True)

    async def store(self, file: UploadFile, name: str):
        """
        Store a file in the storage folder with the given name.
        """
        if not name.endswith(".csv"):
            raise ValueError("The file is not a valid CSV.")

        destination = os.path.join(self.storage_folder, name)
        try:
            # Read and verify the file content
            content = await file.read()
            pd.read_csv(BytesIO(content))

            # Write the file to the storage folder
            with open(destination, 'wb') as dst:
                dst.write(content)
        except Exception as e:
            raise ValueError(f"Failed to verify or store CSV file: {e}")
        finally:
            await file.close()

        return destination

    def load(self, name: str):
        """
        Load and parse a CSV file as a Pandas DataFrame.
        """
        file_path = os.path.join(self.storage_folder, name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{name}' does not exist in the storage folder.")

        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Failed to load CSV file: {e}")
