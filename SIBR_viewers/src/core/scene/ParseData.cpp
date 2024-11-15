/*
 * Copyright (C) 2020, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr
 */



#include "ParseData.hpp"

#include <fstream>
#include <sstream>


#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/case_conv.hpp>
#include <map>
#include "core/system/String.hpp"
#include "core/graphics/Mesh.hpp"

using namespace boost::algorithm;
namespace sibr {


	bool ParseData::parseBundlerFile(const std::string & bundler_file_path)
	{
		// check bundler file
		std::ifstream bundle_file(bundler_file_path);
		if (!bundle_file.is_open()) {
			SIBR_ERR << "Bundler file does not exist at " + bundler_file_path << std::endl;
		}

		// read number of images
		std::string line;
		getline(bundle_file, line);	// ignore first line - contains version

		bundle_file >> _numCameras;	// read first value (number of images)
		getline(bundle_file, line);	// ignore the rest of the line

		//_outputCamsMatrix.resize(_numCameras);
		_camInfos.resize(_numCameras);
		for (int i = 0; i < _numCameras; i++) {
			const sibr::ImageListFile::Infos& infos = _imgInfos[i];

			//Matrix4f &m = _outputCamsMatrix[i];
			Matrix4f m;
			bundle_file >> m(0) >> m(1) >> m(2) >> m(3) >> m(4);
			bundle_file >> m(5) >> m(6) >> m(7) >> m(8) >> m(9);
			bundle_file >> m(10) >> m(11) >> m(12) >> m(13) >> m(14);

			_camInfos[i] = InputCamera::Ptr(new InputCamera(infos.camId, infos.width, infos.height, m, _activeImages[i]));
			_camInfos[i]->name(infos.filename);
			_camInfos[i]->znear(0.001f);
			_camInfos[i]->zfar(1000.0f);
		}

		return true;
	}

	void ParseData::populateFromCamInfos()
	{
		_numCameras = _camInfos.size();
		_imgInfos.resize(_numCameras);
		_activeImages.resize(_numCameras);
		for (uint id = 0; id < _numCameras; id++) {
			_imgInfos[id].camId = _camInfos[id]->id();
			_imgInfos[id].filename = _camInfos[id]->name();
			_imgInfos[id].height = _camInfos[id]->h();
			_imgInfos[id].width = _camInfos[id]->w();

			_activeImages[id] = _camInfos[id]->isActive();
		}
	}

	bool ParseData::parseSceneMetadata(const std::string& scene_metadata_path)
	{

		std::string line;
		std::vector<std::string> splitS;
		std::ifstream scene_metadata(scene_metadata_path);
		if (!scene_metadata.is_open()) {
			return false;
		}

		uint camId = 0;
		while (getline(scene_metadata, line))
		{
			//std::cout << line << '\n';
			if (line.compare("[list_images]") == 0) {
				getline(scene_metadata, line);	// ignore template specification line
				ImageListFile::Infos infos;
				int id;
				while (getline(scene_metadata, line))
				{
					//std::cout << line << std::endl;
					split(splitS, line, is_any_of(" "));
					//std::cout << splitS.size() << std::endl;
					if (splitS.size() > 1) {
						infos.filename = splitS[0];
						infos.width = stoi(splitS[1]);
						infos.height = stoi(splitS[2]);
						infos.camId = camId;

						//infos.filename.erase(infos.filename.find_last_of("."), std::string::npos);
						id = atoi(infos.filename.c_str());

						InputCamera::Z nearFar(100.0f, 0.1f);

						if (splitS.size() > 3) {
							nearFar.near = stof(splitS[3]);
							nearFar.far = stof(splitS[4]);
						}
						_imgInfos.push_back(infos);

						++camId;
						infos.filename.clear();
						splitS.clear();
					}
					else
						break;
				}
			}
			else if (line.compare("[active_images]") == 0) {

				getline(scene_metadata, line);	// ignore template specification line

				_activeImages.resize(_imgInfos.size());

				for (int i = 0; i < _imgInfos.size(); i++)
					_activeImages[i] = false;

				while (getline(scene_metadata, line))
				{
					split(splitS, line, is_any_of(" "));
					//std::cout << splitS.size() << std::endl;
					if (splitS.size() >= 1) {
						for (auto& s : splitS)
							if (!s.empty())
								_activeImages[stoi(s)] = true;
						splitS.clear();
						break;
					}
					else
						break;
				}
			}
			else if (line.compare("[exclude_images]") == 0) {

				getline(scene_metadata, line);	// ignore template specification line

				_activeImages.resize(_imgInfos.size());

				for (int i = 0; i < _imgInfos.size(); i++)
					_activeImages[i] = true;

				while (getline(scene_metadata, line))
				{
					split(splitS, line, is_any_of(" "));
					if (splitS.size() >= 1) {
						for (auto& s : splitS)
							if (!s.empty())
								_activeImages[stoi(s)] = false;
						splitS.clear();
						break;
					}
					else
						break;
				}
			}
			else if (line == "[proxy]") {
				// Read the relative path of the mesh to load.
				getline(scene_metadata, line);
				_meshPath = _basePathName + "/" + line;
			}
		}

		if (_activeImages.empty()) {
			_activeImages.resize(_imgInfos.size());
			for (int i = 0; i < _imgInfos.size(); i++) {
				_activeImages[i] = true;
			}
		}



		scene_metadata.close();

		return true;
	}

	void ParseData::getParsedBundlerData(const std::string & dataset_path, const std::string & customPath, const std::string & scene_metadata_filename)
	{
		_basePathName = dataset_path + customPath;
		/*std::cout << scene_metadata_filename << std::endl;*/
		if (!parseSceneMetadata(_basePathName + "/" + scene_metadata_filename)) {
			SIBR_ERR << "Scene Metadata file does not exist at /" + _basePathName + "/." << std::endl;
		}

		if (!parseBundlerFile(_basePathName + "/cameras/bundle.out")) {
			SIBR_ERR << "Bundle file does not exist at /" + _basePathName + "/cameras/." << std::endl;
		}

		_imgPath = _basePathName + "/images/";

		// Default mesh path if none found in the metadata file.
		if (_meshPath.empty()) {
			_meshPath = _basePathName + "/meshes/recon.obj";
			_meshPath = (sibr::fileExists(_meshPath)) ? _meshPath : _basePathName + "/meshes/recon.ply";
		}

	}

	void ParseData::getParsedMeshroomData(const std::string & dataset_path, const std::string & customPath)
	{		
		_basePathName = dataset_path;

		std::string meshRoomCachePath = sibr::listSubdirectories(_basePathName + "/StructureFromMotion/")[0];

		_camInfos = sibr::InputCamera::loadMeshroom(_basePathName + "/StructureFromMotion/" + meshRoomCachePath);

		if (_camInfos.empty()) {
			SIBR_ERR << "Could not load Meshroom sfm file at /" + _basePathName + "/StructureFromMotion/"<< meshRoomCachePath << std::endl;
		}

		_imgPath = _basePathName + "/PrepareDenseScene/" + sibr::listSubdirectories(_basePathName + "/PrepareDenseScene/")[0];

		populateFromCamInfos();

		_meshPath = _basePathName + "/Texturing/" + sibr::listSubdirectories(_basePathName + "/Texturing/")[0] + "/texturedMesh.obj";

	}

	void ParseData::getParsedColmapData(const std::string & dataset_path, const int fovXfovY_flag, const bool capreal_flag)
	{
		_basePathName = dataset_path + "/colmap/stereo";

		_camInfos = sibr::InputCamera::loadColmap(_basePathName + "/sparse", 0.01f, 1000.0f, fovXfovY_flag);

		if (_camInfos.empty()) {
			SIBR_ERR << "Colmap camera calibration file does not exist at /" + _basePathName + "/sparse/." << std::endl;
		}

		_imgPath = _basePathName + "/images/";

		std::string blackListFile = dataset_path + "/colmap/database.blacklist";

		if (sibr::fileExists(blackListFile)) {
			std::string line;
			std::vector<std::string> splitS;
			std::ifstream blackListFile(blackListFile);
			if (blackListFile.is_open()) {
				while (std::getline(blackListFile, line)) {
					split(splitS, line, is_any_of(" "));
					//std::cout << splitS.size() << std::endl;
					if (splitS.size() > 0) {
						for (uint cam_id = 0; cam_id < _camInfos.size(); cam_id++) {
							if (find_any(splitS, _camInfos[cam_id]->name())) {
								_camInfos[cam_id]->setActive(false);
							}
						}
						splitS.clear();
					}
					else
						break;
				}
			}
		}

		populateFromCamInfos();

		if(capreal_flag) {
			_meshPath = dataset_path + "/capreal/mesh.obj";
			_meshPath = (sibr::fileExists(_meshPath)) ? _meshPath : dataset_path + "/capreal/mesh.ply";
		}
		else {
			_meshPath = dataset_path + "/colmap/stereo/meshed-delaunay.ply";
		}

	}

	void ParseData::getParsedNVMData(const std::string & dataset_path, const std::string & customPath, const std::string & nvm_path)
	{
		_basePathName = dataset_path + customPath + nvm_path;

		_camInfos = sibr::InputCamera::loadNVM(_basePathName + "/scene.nvm", 0.001f, 1000.0f);
		if (_camInfos.empty()) {
			SIBR_ERR << "Error reading NVM dataset at /" + _basePathName << std::endl;
		}

		_imgPath = _basePathName;

		populateFromCamInfos();

		_meshPath = dataset_path + "/capreal/mesh.obj";
		_meshPath = (sibr::fileExists(_meshPath)) ? _meshPath : dataset_path + "/capreal/mesh.ply";
	}

	void ParseData::getParsedData(const BasicIBRAppArgs & myArgs, const std::string & customPath)
	{
		std::string datasetTypeStr = myArgs.dataset_type.get();
		
		boost::algorithm::to_lower(datasetTypeStr);

		std::string bundler = myArgs.dataset_path.get() + customPath + "/cameras/bundle.out";
		std::string colmap = myArgs.dataset_path.get() + "/colmap/stereo/sparse/images.txt";
		std::string caprealobj = myArgs.dataset_path.get() + "/capreal/mesh.obj";
		std::string caprealply = myArgs.dataset_path.get() + "/capreal/mesh.ply";
		std::string nvmscene = myArgs.dataset_path.get() + customPath + "/nvm/scene.nvm";
		std::string meshroom = myArgs.dataset_path.get() + "/../../StructureFromMotion/";
		std::string meshroom_sibr = myArgs.dataset_path.get() + "/StructureFromMotion/";

		if(datasetTypeStr == "sibr") {
			if (!sibr::fileExists(bundler))
				SIBR_ERR << "Cannot use dataset_type " + myArgs.dataset_type.get() + " at /" + myArgs.dataset_path.get() + "." << std::endl
						 << "Reason : bundler folder (" << bundler << ") does not exist" << std::endl;

			_datasetType = Type::SIBR;
		}
		else if (datasetTypeStr == "colmap_capreal") {
			if (!sibr::fileExists(colmap))
				SIBR_ERR << "Cannot use dataset_type " + myArgs.dataset_type.get() + " at /" + myArgs.dataset_path.get() + "." << std::endl
						 << "Reason : colmap folder (" << colmap << ") does not exist" << std::endl;
			
			if (!(sibr::fileExists(caprealobj) || sibr::fileExists(caprealply)))
				SIBR_ERR << "Cannot use dataset_type " + myArgs.dataset_type.get() + " at /" + myArgs.dataset_path.get() + "." << std::endl
						 << "Reason : capreal mesh (" << caprealobj << ", " << caprealply << ") does not exist" << std::endl;

			_datasetType = Type::COLMAP_CAPREAL;
		}
		else if (datasetTypeStr == "colmap") {
			if (!sibr::fileExists(colmap))
				SIBR_ERR << "Cannot use dataset_type " + myArgs.dataset_type.get() + " at /" + myArgs.dataset_path.get() + "." << std::endl
						 << "Reason : colmap folder (" << colmap << ") does not exist" << std::endl;

			_datasetType = Type::COLMAP;
		}
		else if (datasetTypeStr == "nvm") {
			if (!sibr::fileExists(nvmscene))
				SIBR_ERR << "Cannot use dataset_type " + myArgs.dataset_type.get() + " at /" + myArgs.dataset_path.get() + "." << std::endl
						 << "Reason : nvmscene folder (" << nvmscene << ") does not exist" << std::endl;

			_datasetType = Type::NVM;
		}
		else if (datasetTypeStr == "meshroom") {
			if (!(sibr::directoryExists(meshroom) || sibr::directoryExists(meshroom_sibr)))
				SIBR_ERR << "Cannot use dataset_type " + myArgs.dataset_type.get() + " at /" + myArgs.dataset_path.get() + "." << std::endl
						 << "Reason : meshroom folder (" << meshroom << ", " << meshroom_sibr << ") does not exist" << std::endl;

			_datasetType = Type::MESHROOM;
		}
		else {
			if (sibr::fileExists(bundler)) {
				_datasetType = Type::SIBR;
			}
			else if (sibr::fileExists(colmap) && (sibr::fileExists(caprealobj) || sibr::fileExists(caprealply))) {
				_datasetType = Type::COLMAP_CAPREAL;
			}
			else if (sibr::fileExists(colmap)) {
				_datasetType = Type::COLMAP;
			}
			else if (sibr::fileExists(nvmscene)) {
				_datasetType = Type::NVM;
			}
			else if (sibr::directoryExists(meshroom) || sibr::directoryExists(meshroom_sibr)) {
				_datasetType = Type::MESHROOM;
			}
			else {
				SIBR_ERR << "Cannot determine type of dataset at /" + myArgs.dataset_path.get() + customPath << std::endl;
			}
		}

		switch(_datasetType) {
			case Type::SIBR : 			getParsedBundlerData(myArgs.dataset_path, customPath, myArgs.scene_metadata_filename); break;
			case Type::COLMAP_CAPREAL : getParsedColmapData(myArgs.dataset_path, myArgs.colmap_fovXfovY_flag, true); break;
			case Type::COLMAP : 		getParsedColmapData(myArgs.dataset_path, myArgs.colmap_fovXfovY_flag, false); break;
			case Type::NVM : 			getParsedNVMData(myArgs.dataset_path, customPath, "/nvm/"); break;
			case Type::MESHROOM : 		if (sibr::directoryExists(meshroom)) getParsedMeshroomData(myArgs.dataset_path.get() + "/../../");
										else if (sibr::directoryExists(meshroom_sibr)) getParsedMeshroomData(myArgs.dataset_path); break;
		}
		
		// What happens if multiple are present?
		// Ans: Priority --> SIBR > COLMAP > NVM
		
		// Find max cam ID and check present image IDs
		int maxId = 0;
		std::vector<bool> presentIDs;
		
		presentIDs.resize(_numCameras);

		for (int c = 0; c < _numCameras; c++) {
			maxId = (maxId > int(_imgInfos[c].camId)) ? maxId : int(_imgInfos[c].camId);
			try
			{
				presentIDs[_imgInfos[c].camId] = true;
			}
			catch (const std::exception&)
			{
				SIBR_ERR << "Incorrect Camera IDs " << std::endl;
			}
		}

		// Check if max cam ID matches max number of cams
		// If not find the missing IDs 
		std::vector<int> missingIDs;
		int curid;
		int j, pos;
		if (maxId >= _numCameras) {
			for (int i = 0; i < _numCameras; i++) {
				if (!presentIDs[i]) { missingIDs.push_back(i); }
			}

			// Now, shift the imgInfo IDs to adjust max Cam IDs
			for (int k = 0; k < _numCameras; k++) {
				curid = _imgInfos[k].camId;
				pos = -1;
				for (j = 0; j < missingIDs.size(); j++) {
					if (curid > missingIDs[j]) { pos = j; }
					else { break; }
				}

				_imgInfos[k].camId = _imgInfos[k].camId - (pos + 1);
			}
		}

	}

}
