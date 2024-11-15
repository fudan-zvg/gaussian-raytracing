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


#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>
#include "core/assets/Resources.hpp"

/// \todo TODO: If you care about security (did someone want to hack/use your app
/// to hide a virus/retrieve informations from this compiled code), comment
/// the following line and resolve warnings by finding new safe-functions.
#pragma warning(disable:4996) // affect this .cpp only

namespace sibr
{

	Resources* Resources::_instance = NULL;

	Resources* Resources::Instance()
	{
		if (_instance == 0)
			_instance = new Resources;
		return _instance;
	}

	Resources::Resources()
	{
		_rscPaths.push_back(sibr::getInstallDirectory());
		std::ifstream rscFile(sibr::getInstallDirectory() + "/ibr_resources.ini");
		if(rscFile.good())
		{
			for(std::string line; std::getline(rscFile, line); )
			{
				_rscPaths.push_back(line);
			}
		}

		/// \todo WIP: used in prevision to load plugins (TODO: test under linux)
		std::ifstream pathFile(sibr::getInstallDirectory() + "/ibr_paths.ini");
		if(pathFile.good())
		{
			for(std::string line; std::getline(pathFile, line); )
			{
				std::string name    = line.substr(0, line.find("="));
				std::string value   = line.substr(line.find("=")+1, line.length());
				char* curEnv = getenv(name.c_str());
				std::string currentEnv;
				if(curEnv!=NULL)
					currentEnv = std::string(curEnv);
#ifdef _WIN32
				std::replace(value.begin(), value.end(), '/', '\\'); // linux to windows path
				char delimiter = ';';
#else
				std::replace(value.begin(), value.end(), '\\', '/'); // windows to linux path
				char delimiter = ':';
#endif
				std::stringstream ss;
				ss << delimiter;
				if(!currentEnv.empty())
					if (currentEnv.at(currentEnv.length()-1) != delimiter)
						currentEnv.append(ss.str());    

				line = name + "=" + currentEnv + value;
				putenv(const_cast<char*>(line.c_str()));

				std::cout<<"[Resources] env: "<<name<<"="<<getenv(name.c_str())<<std::endl;
			}
		}
	}

	Resources::~Resources()
	{
	}

	std::string Resources::getResourceFilePathName(std::string const & filename, bool & success)
	{
		// we assume the first element of _rscPaths if the current dir
		for(std::string rscPath : _rscPaths)
		{
			std::string filePathName  = sibr::getInstallDirectory() + "/" + rscPath + "/" + filename;
			std::ifstream rscFile(filePathName);
			if (success = rscFile.good()) {
				return filePathName;
			}
		}
		return filename;
	}

	std::string Resources::getResourceFilePathName(std::string const & filename)
	{
		bool success = false;
		return getResourceFilePathName(filename,success);
	}

} // namespace sibr
