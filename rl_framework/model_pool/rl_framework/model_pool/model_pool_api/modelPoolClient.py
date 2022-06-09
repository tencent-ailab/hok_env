import json
from collections import namedtuple
from urllib.parse import urlparse
import requests


ModelFile = namedtuple(
    "ModelFile", "fid filename extraKey size timestampNano absPath customData"
)
GetFileInfoResponse = namedtuple("GetFileInfoResponse", "currentFileInModelpool files")
UploadResponse = namedtuple("UploadResponse", "fid")


class ModelPoolClient:
    def __init__(self, targetAddress):
        if targetAddress.find("://") < 0:
            targetAddress = "http://" + targetAddress
        url = urlparse(targetAddress)
        url = url._replace(scheme="http")
        self._targetURL = url

    def getFileInfo(
        self, fid=None, filename=None, extraKey=None, newest=None, hasExtraKey=None
    ):
        req = {}
        req["fid"] = fid
        req["filename"] = filename
        req["extraKey"] = extraKey
        req["newest"] = newest
        req["hasExtraKey"] = hasExtraKey
        reqJson = json.dumps(req)
        url = self._targetURL._replace(path="get_file_info")
        r = requests.post(url.geturl(), json=req)
        rsp = r.json()
        if "currentFileInModelpool" not in rsp or rsp["currentFileInModelpool"] == 0:
            return GetFileInfoResponse(0, [])
        files = []
        for rspFile in rsp["files"]:
            if "filename" in rspFile:
                filename = rspFile["filename"]
            if "extraKey" in rspFile:
                extraKey = rspFile["extraKey"]
            if "size" in rspFile:
                size = rspFile["size"]
                size = int(size)
            if "timestampNano" in rspFile:
                timestampNano = rspFile["timestampNano"]
                timestampNano = int(timestampNano)
            if "absPath" in rspFile:
                absPath = rspFile["absPath"]
            if "customData" in rspFile:
                customData = rspFile["customData"]
            else:
                customData = None
            file = ModelFile(
                rspFile["fid"],
                filename,
                extraKey,
                size,
                timestampNano,
                absPath,
                customData,
            )
            files.append(file)
        return GetFileInfoResponse(rsp["currentFileInModelpool"], files)

    def download(self, fd, fid=None, filename=None, extraKey=None):
        url = self._targetURL._replace(path="/download")
        req = {}
        req["fid"] = fid
        req["filename"] = filename
        req["extraKey"] = extraKey
        r = requests.get(url.geturl(), params=req, stream=True)
        for chunk in r.iter_content(chunk_size=(1 << 20)):
            fd.write(chunk)
        if r.status_code != 200:
            raise ConnectionError("response status code is " + str(r.status_code))

    def upload(self, filePath, extraKey=None, customData=None):
        url = self._targetURL._replace(path="/upload")
        req = {}
        req["extraKey"] = extraKey
        req["customData"] = customData
        with open(filePath, "rb") as file:
            r = requests.post(url.geturl(), files={"data": file}, params=req)
        rsp = r.json()
        return UploadResponse(fid=rsp["fid"])

    def uploadBytes(self, fd, extraKey=None, filename=None, customData=None):
        url = self._targetURL._replace(path="/upload_bytes")
        req = {}
        req["filename"] = filename
        req["extraKey"] = extraKey
        req["customData"] = customData
        r = requests.post(url.geturl(), data=fd, params=req)
        rsp = r.json()
        return UploadResponse(fid=rsp["fid"])

    def delete(self, fids=None, extraKeys=None, filenames=None, deleteNoKeys=None):
        url = self._targetURL._replace(path="/delete")
        req = {}
        req["fids"] = fids
        req["filenames"] = filenames
        req["extraKeys"] = extraKeys
        req["deleteNoKeys"] = deleteNoKeys
        r = requests.post(url.geturl(), json=req)
        rsp = r.json()
        if "deleteFids" not in rsp:
            rsp["deleteFids"] = []
        return rsp

    def howAreYou(self):
        url = self._targetURL._replace(path="/how_are_you")
        r = requests.get(url.geturl())
        result = r.json()
        return result

    def heartBeatCheckAll(self):
        url = self._targetURL._replace(path="/heart_beat_check_all")
        requests.get(url.geturl())
