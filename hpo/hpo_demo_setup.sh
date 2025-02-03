#!/bin/bash
#
# Copyright (c) 2020, 2022 Red Hat, IBM Corporation and others.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# include the common_utils.sh script to access methods
current_dir="$(dirname "$0")"
common_dir="${current_dir}/../common/"
source ${common_dir}/common_helper.sh

function usage() {
	echo "Usage: $0 [-s|-t] [-o hpo-image] [-r] [-c cluster-type] [-b benchmark-cluster-type] [-m benchmark-server] [--benchmark=benchmark-name] [--searchspace=searchspace.json] [-j] "
	echo "s = start (default), t = terminate"
	echo "r = restart hpo only"
	echo "c = supports native, docker and Operate-first cluster-type to start HPO service"
	echo "o = hpo image"
	echo "b = cluster on which benchmark runs"
	echo "m = server name on which benchmark is run"
	echo "j = run benchmark on jenkins"
	echo "e = disable hpo experiments"
	echo "benchmark = benchmark to run. Default techempower"
	echo "searchspace = searchspace json"
	echo "jenkinsmachine jenkinsport jenkinsjob jenkinstoken jenkinsrepo jenkinshorreum  :JENKINS CONFIGURATION"
	echo "p = expose prometheus port"
	exit 1
}

## Checks for the pre-requisites to run the demo benchmark with HPO.
function prereq_check() {
	# Python is required only if we're installing the app as a 'Native'
	if [ "$1" == "native" ]; then
		## Requires python3 to start HPO
		python3 --version >/dev/null 2>/dev/null
		check_err "ERROR: python3 not installed. Required to start HPO. Check if all dependencies (python3,minikube,php,java11,wget,curl,zip,bc,jq) are installed."
	fi
	if [[ "$1" == "minikube" ]]; then
		## Requires minikube to run the demo benchmark for experiments
		minikube >/dev/null 2>/dev/null
		check_err "ERROR: minikube not installed. Required for running benchmark. Check if all other dependencies (php,java11,git,wget,curl,zip,bc,jq) are installed."
		## Check version of minikube to be <= 1.26.1 for support of Prometheus version 0.8.0
		minikube_ver=$(minikube version | grep "version" | sed 's/minikube version: v\([0-9]\+\).\([0-9]\+\).\([0-9]\+\).*/\1\2\3/')
		if [ "$minikube_ver" -gt "1261" ]; then
			echo "Current minikube version not supported: $(minikube version)"
			echo "Supported Version 1.26.1 or less";
			exit 1;
		fi
		kubectl get pods >/dev/null 2>/dev/null
		check_err "ERROR: minikube not running. Required for running benchmark"
		## Check if prometheus is running for valid benchmark results.
		prometheus_pod_running=$(kubectl get pods --all-namespaces | grep "prometheus-k8s-0")
		if [ "${prometheus_pod_running}" == "" ]; then
			err_exit "Install prometheus for valid results from benchmark."
		fi
	fi

	if [[ ${BENCHMARK_RUN_THRU} == "standalone" ]]; then
		## Requires java 11
		java -version >/dev/null 2>/dev/null
		check_err "Error: java is not found. Requires Java 11 for running benchmark."
		JAVA_VERSION=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}')
		if [[ ${JAVA_VERSION} < "11" ]]; then
			err_exit "ERROR: Java 11 is required."
		fi
		## Requires wget
		wget --version >/dev/null 2>/dev/null
		check_err "ERROR: wget not installed. Required for running benchmark. Check if all other dependencies (php,curl,zip,bc,jq) are installed."
		## Requires curl
		curl --version >/dev/null 2>/dev/null
		check_err "ERROR: curl not installed. Required for running benchmark. Check if all other dependencies (php,zip,bc,jq) are installed."
		## Requires bc
		bc --version >/dev/null 2>/dev/null
		check_err "ERROR: bc not installed. Required for running benchmark. Check if all other dependencies (php,zip,jq) are installed."
		## Requires jq
		jq --version >/dev/null 2>/dev/null
		check_err "ERROR: jq not installed. Required for running benchmark. Check if all other dependencies (php,zip) are installed."
		## Requires zip
		zip --version >/dev/null 2>/dev/null
		check_err "ERROR: zip not installed. Required for running benchmark. Check if other dependencies (php) are installed."
		## Requires php
		php --version >/dev/null 2>/dev/null
		check_err "ERROR: php not installed. Required for running benchmark."
	fi

}

###########################################
#   Start HPO
###########################################
function hpo_install() {
	echo
	echo "#######################################"
	echo "Start HPO Server"
	if [ ! -d hpo ]; then
		echo "ERROR: hpo dir not found."
		if [ ${hpo_restart} -eq 1 ]; then
			echo "ERROR: HPO not running. Wrong use of restart command"
		fi
		exit -1
	fi
	pushd hpo >/dev/null
		if [ -z "${HPO_DOCKER_IMAGE}" ]; then
			HPO_VERSION=$(cat version.py | grep "HPO_VERSION" | cut -d "=" -f2 | tr -d '"')
			HPO_DOCKER_IMAGE=${HPO_DOCKER_REPO}:${HPO_VERSION}
		fi
		if [[ ${hpo_restart} -eq 1 ]]; then
			echo
			echo "Terminating the HPO server"
			echo
			./deploy_hpo.sh -c ${CLUSTER_TYPE} -t
			check_err "ERROR: HPO failed to terminate, exiting"
		fi
		if [[ ${CLUSTER_TYPE} == "native" ]]; then
			echo
			echo "Terminating before starting"
                        SERVICE_STATUS_NATIVE=$(ps -ef | grep service.py | grep -v grep | awk '{print $2}')
                        echo "Before SERVICE_STATUS_NATIVE= ${SERVICE_STATUS_NATIVE}"
                        ps -ef | grep src/service.py | grep -v grep | awk '{print $2}' | xargs kill -9 >/dev/null 2>&1

			echo "Starting hpo with  ./deploy_hpo.sh -c ${CLUSTER_TYPE} -p 8092 --rest"
			echo
			./deploy_hpo.sh -c ${CLUSTER_TYPE} -p 8092 --rest >> ${LOGFILE} 2>&1 &
			check_err "ERROR: HPO failed to start, exiting"
		else
			echo
			echo "Starting hpo with  ./deploy_hpo.sh -c ${CLUSTER_TYPE} -o ${HPO_DOCKER_IMAGE}"
			echo

			./deploy_hpo.sh -c "${CLUSTER_TYPE}" -o "${HPO_DOCKER_IMAGE}" >> ${LOGFILE} 2>&1
			check_err "ERROR: HPO failed to start, exiting"
		fi
	popd >/dev/null
	echo "#######################################"
	echo

}

# Function to get the URL to access HPO
function getURL() {
	if [[ ${CLUSTER_TYPE} == "operate-first" ]]; then
		url="http://hpo-openshift-tuning.apps.smaug.na.operate-first.cloud"
	else
		if [[ ${CLUSTER_TYPE} == "native" ]] || [[ ${CLUSTER_TYPE} == "docker" ]]; then
			service_msg="Access REST Service at"
		else
			service_msg="Access HPO at"
		fi
		url=`awk '/'"${service_msg}"'/{print $NF}' "${LOGFILE}" | tail -1`
	fi
	echo "${url}"
}

###########################################
#   Start HPO Experiments
###########################################
## This function starts the experiments with the provided searchspace json.
## It can be customized to run for any usecase by
## 1. Providing the searchspace json
## 2. Modifying "Step 3" to run the usecase specific benchmark
## Currently, it uses TechEmpower benchmark running in minikube for the demo.
function hpo_experiments() {

	URL=$(getURL)
	exp_json=$(cat ${SEARCHSPACE_JSON})
	if [[ ${exp_json} == "" ]]; then
		err_exit "Error: Searchspace is empty"
	fi
	## Get experiment_name from searchspace
	ename=$(${PY_CMD} -c "import hpo_helpers.utils; hpo_helpers.utils.getexperimentname(\"${SEARCHSPACE_JSON}\")")
	## Get total_trials from searchspace
	ttrials=$(${PY_CMD} -c "import hpo_helpers.utils; hpo_helpers.utils.gettrials(\"${SEARCHSPACE_JSON}\")")
	if [[ ${ename} == "" || ${ttrials} == "" ]]; then
		err_exit "Error: Invalid search space"
	fi

	echo "#######################################"
	echo "Start a new experiment with search space json"
	## Step 1 : Start a new experiment with provided search space.
	echo "curl -o response.txt -w "%{http_code}" -H 'Content-Type: application/json' ${URL}/experiment_trials -d '{ "operation": "EXP_TRIAL_GENERATE_NEW",  "search_space": '"${exp_json}"'}'"
	http_response=$(curl -o response.txt -w "%{http_code}" -H 'Content-Type: application/json' ${URL}/experiment_trials -d '{ "operation": "EXP_TRIAL_GENERATE_NEW",  "search_space": '"${exp_json}"'}')
	if [ "$http_response" != "200" ]; then
		err_exit "Error:" $(cat response.txt)
	fi

	## Looping through trials of an experiment
	echo
	echo "Starting an experiment with ${ttrials} trials to optimize ${BENCHMARK_NAME}"
	echo
	for (( i=0 ; i<${ttrials} ; i++ ))
	do
		## Step 2: Get the HPO config from HPOaaS
		echo "#######################################"
		echo
		echo "Generate the config for trial ${i}"
		echo
		sleep 5
		HPO_CONFIG=$(curl -LfSs -H 'Accept: application/json' "${URL}"'/experiment_trials?experiment_name='"${ename}"'&trial_number='"${i}")
		check_err "Error: Issue generating the configuration from HPO."
		echo "${HPO_CONFIG}" | tee hpo_config.json

		## Step 3: Run the benchmark with HPO config.
		## Output of the benchmark should contain objective function result value and status of the benchmark.
		## Status of the benchmark supported is success and failure
		## Output format expected for BENCHMARK_OUTPUT is "Objfunc_result=0.007914818407446147 Benchmark_status=success"
		## Status of benchmark trial is set to failure, if objective function result value is not a number.
		echo "#######################################"
		echo
		echo "Run the benchmark for trial ${i}"
		echo
		BENCHMARK_OUTPUT=$(./hpo_helpers/runbenchmark.sh "hpo_config.json" "${SEARCHSPACE_JSON}" "$i" "${BENCHMARK_CLUSTER}" "${BENCHMARK_SERVER}" "${BENCHMARK_NAME}" "${BENCHMARK_RUN_THRU}" "${JENKINS_MACHINE_NAME}" "${JENKINS_EXPOSED_PORT}" "${JENKINS_SETUP_JOB}" "${JENKINS_SETUP_TOKEN}" "${JENKINS_GIT_REPO_COMMIT}" "${HORREUM}" | tee /dev/tty)
		echo ${BENCHMARK_OUTPUT}
		obj_result=$(echo ${BENCHMARK_OUTPUT} | awk '{for(i=1;i<=NF;i++) if($i ~ /^Objfunc_result=/) {split($i,a,"="); print a[2]}}')
		trial_state=$(echo ${BENCHMARK_OUTPUT} | awk '{for(i=1;i<=NF;i++) if($i ~ /^Benchmark_status=/) {split($i,a,"="); print a[2]}}')
		### Setting obj_result=0 and trial_state="failure" to contine the experiment if obj_result is nan or trial_state is empty because of any issue with benchmark output.
		number_check='^[0-9,.]+$'
		if ! [[ ${obj_result} =~  ${number_check} ]]; then
			obj_result=0
			trial_state="failure"
		elif [[ ${trial_state} == "" ]]; then
			trial_state="failure"
		fi

		## Only for now: To avoid mising results incase the HPO is aborted
		cat experiment-output.csv

		## Step 4: Send the results of benchmark to HPOaaS
		echo "#######################################"
		echo
		echo "Send the benchmark results for trial ${i}"
		http_response=$(curl -so response.txt -w "%{http_code}" -H 'Content-Type: application/json' ${URL}/experiment_trials -d '{"experiment_name" : "'"${ename}"'", "trial_number": '"${i}"', "trial_result": "'"${trial_state}"'", "result_value_type": "double", "result_value": '"${obj_result}"', "operation" : "EXP_TRIAL_RESULT"}')
		if [ "$http_response" != "200" ]; then
			err_exit "Error:" $(cat response.txt)
		fi
		echo
		sleep 5
		## Step 5 : Generate a subsequent trial
		if (( i < ${ttrial} - 1 )); then
			echo "#######################################"
			echo
			echo "Generate subsequent trial of ${i}"
			http_response=$(curl -so response.txt -w "%{http_code}" -H 'Content-Type: application/json' ${URL}/experiment_trials -d '{"experiment_name" : "'"${ename}"'", "operation" : "EXP_TRIAL_GENERATE_SUBSEQUENT"}')
			if [ "$http_response" != "200" ]; then
				err_exit "Error:" $(cat response.txt)
		fi
			echo
		fi
	done

	## Gather the plots for importance and optimization history
	echo "curl -o ${HPO_RESULTS_DIR}/tunable_importance.html ${URL}/plot?experiment_name=${ename}&type=tunable_importance"
	curl -o ${HPO_RESULTS_DIR}/tunable_importance.html "${URL}/plot?experiment_name=${ename}&type=tunable_importance"
	curl -o ${HPO_RESULTS_DIR}/optimization_history.html "${URL}/plot?experiment_name=${ename}&type=optimization_history"

	echo "#######################################"
	echo
	echo "Experiment complete"
	echo

}

function hpo_start() {

	if [[ ${CLUSTER_TYPE} == "minikube" ]]; then
		minikube >/dev/null
		check_err "ERROR: minikube not installed"
	fi
	# Start all the installs
	start_time=$(get_date)
	echo
	echo "#######################################"
	echo "#           HPOaaS Demo               #"
	echo "#######################################"
	echo
	echo "--> Starts HPOaaS"
	echo "--> Runs ${BENCHMARK_NAME} benchmark on ${BENCHMARK_CLUSTER}"
	echo "--> Optimizes ${BENCHMARK_NAME} benchmark based on the provided search_space(${SEARCHSPACE_JSON}) using HPOaaS"
	echo "--> search_space provides a performance objective and tunables along with ranges"
	echo

	if [ ${hpo_restart} -eq 0 ]; then
		if [[ ${CLUSTER_TYPE} == "minikube" ]]; then # || [[ ${CLUSTER_TYPE} == "native" ]]; then
			minikube_start
			prometheus_install
		fi
		clone_repos hpo
# clone autotune repo as well to install the prometheus
		if [ ${hpo_experiments} -eq 1 ]; then
			clone_repos autotune
			clone_repos benchmarks
			#benchmarks_install
		fi
	fi
#	Check for pre-requisites to run the demo benchmark with HPO.
	prereq_check ${CLUSTER_TYPE}
#  HPO is already running on operate-first. So, no need to install again.
	if [[ ${CLUSTER_TYPE} != "operate-first" ]]; then
		 # Installing jsonschema explicitly to debug errors
		python3 -m pip install --user --no-cache-dir --force-reinstall optuna
		python3 -m pip install --user --no-cache-dir --force-reinstall requests
		python3 -m pip install --user --no-cache-dir --force-reinstall scikit-optimize
		python3 -m pip install --user --no-cache-dir --force-reinstall jsonschema
		python3 -m pip install --user --no-cache-dir --force-reinstall grpcio
		python3 -m pip install --user --no-cache-dir --force-reinstall click
		python3 -m pip install --user --no-cache-dir --force-reinstall protobuf
		python3 -m pip install --user --no-cache-dir --force-reinstall plotly
		python3 -m pip install --user --no-cache-dir --force-reinstall pandas
		hpo_install
		sleep 10
		cat ${LOGFILE}
	fi

	if [ ${hpo_experiments} -eq 1 ]; then
		hpo_experiments
	fi

	echo
	end_time=$(get_date)
	elapsed_time=$(time_diff "${start_time}" "${end_time}")
	echo "Success! HPO demo setup took ${elapsed_time} seconds"
	echo
	echo "Look into experiment-output.csv for configuration and results of all trials"
	echo "and benchmark.log for demo benchmark logs"
	echo
	if [ ${prometheus} -eq 1 ]; then
		expose_prometheus
	fi

}

function hpo_terminate() {
	echo
	echo "#######################################"
	echo "#       HPO Demo Terminate       #"
	echo "#######################################"
	echo
	pushd hpo >/dev/null
		./deploy_hpo.sh -t -c ${CLUSTER_TYPE}
		#check_err "ERROR: Failed to terminate hpo"
		echo
	popd >/dev/null
}

function hpo_cleanup() {

	delete_repos hpo
	delete_repos autotune
	#minikube_delete
	## Delete the logs if any before starting the experiment
	rm -rf experiment-output.csv hpo_config.json benchmark.log hpo.log response.txt
	echo "Success! HPO demo cleanup completed."
	echo
}

# Default docker image repos
HPO_DOCKER_REPO="quay.io/kruize/hpo"
PY_CMD="python3"
LOGFILE="${PWD}/hpo.log"
HPO_RESULTS_DIR="${PWD}/results"

if [ ! -d "${HPO_RESULTS_DIR}" ]; then
  mkdir -p ${HPO_RESULTS_DIR}
fi

export N_TRIALS=3
export N_JOBS=1
export APP_NAMESPACE="default"

# Default cluster
CLUSTER_TYPE="native"
# Default duration of benchmark warmup/measurement cycles in seconds.
DURATION=60
BENCHMARK_CLUSTER="minikube"
BENCHMARK_SERVER="localhost"
BENCHMARK_RUN_THRU="standalone"
BENCHMARK_NAME="techempower"
SEARCHSPACE_JSON="hpo_helpers/tfb_qrh_search_space.json"

# By default we start the demo & experiment and we dont expose prometheus port
prometheus=0
hpo_restart=0
hpo_experiments=1
start_demo=1

# Iterate through the commandline options
while getopts b:c:d:m:o:ejprstk:-: gopts
do
        case "${gopts}" in
                 -)
                case "${OPTARG}" in
                        jenkinsmachine=*)
                                JENKINS_MACHINE_NAME=${OPTARG#*=}
                                ;;
                        jenkinsport=*)
                                JENKINS_EXPOSED_PORT=${OPTARG#*=}
                                ;;
                        jenkinsjob=*)
                                JENKINS_SETUP_JOB=${OPTARG#*=}
                                ;;
                        jenkinstoken=*)
                                JENKINS_SETUP_TOKEN=${OPTARG#*=}
                                ;;
                        jenkinsrepo=*)
                                JENKINS_GIT_REPO_COMMIT=${OPTARG#*=}
                                ;;
                        benchmark=*)
                                BENCHMARK_NAME=${OPTARG#*=}
                                ;;
			searchspace=*)
				SEARCHSPACE_JSON=${OPTARG#*=}
				;;
			jenkinshorreum=*)
				HORREUM=${OPTARG#*=}
				;;
                        *)
                                ;;
                esac
                ;;
		o)
			HPO_DOCKER_IMAGE="${OPTARG}"
			;;
		p)
			prometheus=1
			;;
		r)
			hpo_restart=1
			;;
		s)
			start_demo=1
			;;
		t)
			start_demo=0
			;;
		e)
			hpo_experiments=0
			;;
		c)
			CLUSTER_TYPE="${OPTARG}"
			;;
		d)
			DURATION="${OPTARG}"
			;;
		b)
			BENCHMARK_CLUSTER="${OPTARG}"
			;;
		m)
			BENCHMARK_SERVER="${OPTARG}"
			;;
		j)
			BENCHMARK_RUN_THRU="jenkins"
			;;
		*)
			usage
	esac
done

if [ ${start_demo} -eq 1 ]; then
	hpo_start
else
	if [[ ${CLUSTER_TYPE} != "operate-first" ]]; then
		hpo_terminate
	fi
	hpo_cleanup
fi
