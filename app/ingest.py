import os

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".doc", ".docx", ".ppt", ".pptx")


def get_document_loader(file_path):
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        return PyPDFLoader(file_path)

    if extension == ".txt":
        from langchain_community.document_loaders import TextLoader

        return TextLoader(file_path)

    if extension == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        return Docx2txtLoader(file_path)

    if extension == ".doc":
        from langchain_community.document_loaders import UnstructuredWordDocumentLoader

        return UnstructuredWordDocumentLoader(file_path)

    if extension in (".ppt", ".pptx"):
        from langchain_community.document_loaders import UnstructuredPowerPointLoader

        return UnstructuredPowerPointLoader(file_path)

    return None


def load_documents(folder_path="documents"):
    """
    Load all supported documents from the documents folder
    """

    documents = []

    # iterate over files in folder
    for file in os.listdir(folder_path):

        file_path = os.path.join(folder_path, file)

        if not os.path.isfile(file_path):
            continue

        loader = get_document_loader(file_path)

        if loader is None:
            continue

        docs = loader.load()
        documents.extend(docs)

    return documents