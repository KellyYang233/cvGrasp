#include "Classifier.hpp"



float sigmoid_predict(float decision_value, float A, float B)
{
	float fApB = decision_value*A+B;
	// 1-p used later; avoid catastrophic cancellation
	if (fApB >= 0)
		return exp(-fApB)/(1.0+exp(-fApB));
	else
		return 1.0/(1+exp(fApB)) ;
}

Mat classificationToConfusionMatrix(const Mat &classifications, const Mat &labels)
{
    int maxY, maxX;
    int maxLabel = matrixMaxValue<int>(labels, maxY, maxX);
    cout << "maxLabel: " << maxLabel << endl;
    Mat confusion = Mat::zeros(maxLabel+1, maxLabel+1, CV_32F);
    int n = classifications.rows;
    for(int i = 0; i < n; i++)
    {
        int lgt = labels.at<int>(i,0);
        int lc, dummy;
        matrixMaxValue<float>(classifications.rowRange(i,i+1), dummy, lc);
        confusion.at<float>(lgt,lc) += 1.0;
    }

	for(int r = 0; r < confusion.rows; r++)
	{
		for(int c = 0; c < confusion.cols; c++)
			cout << confusion.at<float>(r,c) << "\t";
		cout << endl;
	}
	float sum = 0;
	Mat CC = Mat::zeros(1, maxLabel+1, CV_32F);
	for(int c = 0; c < maxLabel+1; c++)
	{
		for(int r = 0; r < maxLabel+1; r++)
		{
			CC.at<float>(0, c) += confusion.at<float>(r,c);
		}
		sum += CC.at<float>(0, c);
	}
	CC = CC/sum;
	for(int c = 0; c < maxLabel+1; c++)
		cout << setw(4) << CC.at<float>(0, c) << "\t";
	cout << endl;

    return confusion;
}

Mat LcClassifier::predict(Mat & feature)
{
	return Mat::zeros(feature.rows, 1, CV_32F);
}

void LcSVM::write_libsvm_input(const Mat& data, const Mat& labels, const char * filename)
{
    int m = data.rows;
    int n = data.cols;

    ofstream outfile(filename, ios::out);

    for(int i = 0; i < m; i++)
    {
        outfile << (int)(labels.at<float>(i,0)) << " ";
        for(int j = 0; j < n; j++)
        {
//            if(data.at<float>(i,j) != 0)
                outfile << j+1 << ":" << data.at<float>(i,j) << " ";
        }
        outfile << endl;
    }

    outfile.close();

    return;
}

void LcSVM::read_libsvm_model(const char * filename, int dim)
{
    int nsv, label;
    int n = dim;
    Mat supvec, alpha;
    float b;

    ifstream infile(filename, ios::in);
    if(!infile)
    {
        cout << "[read_libsvm_model]: failed to read model file: " + string(filename) << endl;
        exit(-1);
    }

    /**** format of model file ******
    svm_type c_svc
    kernel_type linear
    nr_class 2
    total_sv 101
    rho -1.05069
    label 1 -1
    probA -1.13745 //only for probability estimation
    probB 0.0462928 //only for probability estimation
    nr_sv 51 50
    SV (4 class)
    +-+-+-+--------------------+
    |1|1|1|                                      |
    |v|v|v|  SVs from class 1            |
    |2|3|4|                                      |
    +-+-+-+--------------------+
    |1|2|2|                                      |
    |v|v|v|  SVs from class 2            |
    |2|3|4|                                      |
    +-+-+-+--------------------+
    |1|2|3|                                      |
    |v|v|v|  SVs from class 3            |
    |3|3|4|                                      |
    +-+-+-+--------------------+
    |1|2|3|                                      |
    |v|v|v|  SVs from class 4            |
    |4|4|4|                                      |
    +-+-+-+--------------------+
    *****************************/
    int limit = 1000000;
    char line[limit];
    infile.getline(line, limit); //svm_type c_svc
    infile.getline(line, limit); //kernel_type linear
    char *tok_kernel = strtok(line, " ");
	tok_kernel = strtok(NULL, " ");
	
	if(0 == strcmp(tok_kernel, "rbf"))
	{
		_kernel_type = RBF;
		infile.getline(line, limit); //gamma
		char * tok_gamma = strtok(line, " ");
		tok_gamma = strtok(NULL, " ");
		_gamma = atof(tok_gamma);
	}
	else if(0 == strcmp(tok_kernel, "linear"))
	{
		_kernel_type = LINEAR;
	}
    infile.getline(line, limit); //nr_class 2
    infile.getline(line, limit); //total_sv
    char * tok_nsv = strtok(line, " ");
    tok_nsv = strtok(NULL, " ");
    nsv = atoi(tok_nsv);
	if(0 == nsv)
	{
		cout << "[read_libsvm_model]WARNING: no support vector!\n";
		return;
	}
    cout << "nsv: " << nsv << endl;

    supvec = Mat::zeros(nsv, n, CV_32F);
    alpha = Mat::zeros(nsv, 1, CV_32F);

    infile.getline(line, limit); //rho
    char * tok_b = strtok(line, " ");
    tok_b = strtok(NULL, " ");
    b = atof(tok_b);
    cout << "b: " << b << endl;

    infile.getline(line, limit); //label
    char * tok_label = strtok(line, " ");
    tok_label = strtok(NULL, " ");
    label = atoi(tok_label);
	
	infile.getline(line, limit); //probA
	char * tok_probA = strtok(line, " ");
	tok_probA = strtok(NULL, " ");
	_probA = atof(tok_probA);

	infile.getline(line, limit); //probB
	char * tok_probB = strtok(line, " ");
	tok_probB = strtok(NULL, " ");
	_probB = atof(tok_probB);
	
    infile.getline(line, limit); //nr_sv
    infile.getline(line, limit); //SV
    for(int i = 0; i < nsv; i++)
    {
        infile.getline(line, limit); //each support vector
        char * tok = strtok(line, " :");
        alpha.at<float>(i,0) = atof(tok);
        int ind;
        float s;
        while(tok != NULL)
        {
            tok = strtok(NULL, " :");
            if(tok == NULL)
                break;
            ind = atoi(tok);
            tok = strtok(NULL, " :");
            if(tok == NULL)
                break;
            s = atof(tok);
            supvec.at<float>(i, ind-1) = s;
        }
    }
    infile.close();

    cout << supvec.rows << " " << supvec.cols << endl;
	
    _b = -b; // format in original model file: weight*x-bias
    _w = Mat::zeros(1, n, CV_32F);
    for(int i = 0; i < nsv; i++)
        for(int j = 0; j < n; j++)
            _w.at<float>(0,j) += supvec.at<float>(i,j)*alpha.at<float>(i,0);

	_SV = Mat::zeros(nsv, n, CV_32F);
	_sv_coef = Mat::zeros(nsv, 1, CV_32F);
	supvec.copyTo(_SV);
	alpha.copyTo(_sv_coef);

    //To deal with label order: -1, 1
    if(label <= 0)
    {
        _w = -_w;
        _b = -_b;
    }

    return;
}

void LcSVM::save(string filename)
{
    cout << "\nClassifier: Saving " << filename << endl;
    FileStorage ofs;
    ofs.open(filename, FileStorage::WRITE);
	ofs << "kernel_type" << _kernel_type;
    ofs << "weight" << _w;
    ofs << "bias" << _b;
	ofs << "probA" << _probA;
	ofs << "probB" << _probB;
	if(RBF == _kernel_type)
	{
		ofs << "SV" << _SV;
		ofs << "sv_coef" << _sv_coef;
		ofs << "gamma" << _gamma;
	}
}

void LcSVM::load(string filename)
{
    cout << "Classifier: Loading " << filename << endl;
    FileStorage ifs;
    ifs.open(filename, FileStorage::READ);
	ifs["kernel_type"] >> _kernel_type;
    ifs["weight"] >> _w;
    ifs["bias"] >> _b;
	ifs["probA"] >> _probA;
	ifs["probB"] >> _probB;
	if(RBF == _kernel_type)
	{
		ifs["SV"] >> _SV;
		ifs["sv_coef"] >> _sv_coef;
		ifs["gamma"] >> _gamma;
	}
}

void LcSVM::train(Mat &feature, Mat &labels, string dataFile, string modelFile)
{
    char params[300];
    params[0] = 0;
    sprintf(params, "-t 0 -g 8 -c 8 -b 1 -q");

    write_libsvm_input(feature, labels, dataFile.c_str());

	double t = double(getTickCount());
    char command[500];
    command[0] = 0;
    sprintf(command, "./libsvm-3.17/svm-train %s %s %s", params, dataFile.c_str(), modelFile.c_str());
    cout << command << endl;
    system(command);
	t = (getTickCount()-t)/getTickFrequency();
	cout << " time to train:" << t << " secs." << endl;

    read_libsvm_model(modelFile.c_str(), feature.cols);

    return;
}

Mat LcSVM::RBF_response(Mat & feature)
{
	if(feature.cols != _SV.cols)
	{
		cout << "[LcSVM::RBF_response] ERROR: unequal feature dimension\n";
		exit(-1);
	}

	Mat output = Mat::zeros(feature.rows, 1, CV_32F);
	for(int i = 0; i < feature.rows; i++)
	{
		float sum_rbf = 0;
		for(int j = 0; j < _SV.rows; j++)
		{
			float sum_diff2 = 0;
			for(int k = 0; k < _SV.cols; k++)
				sum_diff2 += pow(feature.at<float>(i,k)-_SV.at<float>(j,k), 2);
			sum_rbf += _sv_coef.at<float>(j,0) * exp(-_gamma*sum_diff2);
//			cout << "exp(" << -_gamma*sum_diff2 << ") + ";
		}
		sum_rbf += _b;
		output.at<float>(i,0) = sum_rbf;
//		cout << _b << " = " << sum_rbf << endl;
	}

	return output;
}

Mat LcSVM::predict(Mat & feature)
{
    if(0 == _w.rows)
        return Mat::zeros(feature.rows, 1, CV_32F);
	if(LINEAR == _kernel_type)		
    	return feature*_w.t()+_b;
	else if(RBF == _kernel_type)
		return RBF_response(feature);
	else
	{
		cout << "[LcSVM::predict] ERROR: unrecognized kernel type: " << _kernel_type << endl;
		exit(-1);
	}
}

Mat LcSVM::predict_prob(Mat & feature)
{
	if(0 == _w.rows)
		return Mat::ones(feature.rows, 1, CV_32F)*0.1;
	float min_prob=1e-7;
	Mat score = predict(feature);
	Mat prob(score.size(), score.type());
	for(int r = 0; r < score.rows; r++)
		prob.at<float>(r,0) = min(max(sigmoid_predict(score.at<float>(r,0), _probA, _probB),min_prob),1-min_prob);
	return prob;
}

void LcRandomTreesR::save(string filename)
{
    cout << "\nClassifier: Saving " << filename << endl;
	_random_tree.save( filename.c_str());
}

void LcRandomTreesR::load(string filename)
{
    cout << "Classifier: Loading " << filename << endl;
    _random_tree.load( filename.c_str());
	_isTrained = true;
}

void LcRandomTreesR::train(Mat &feature, Mat &labels, string dataFile, string modelFile)
{
	_params.max_depth				= 10;
	_params.regression_accuracy		= 0.00f;
	_params.min_sample_count		= 10;
/*
	_params.calc_var_importance 	= true;
	_params.nactive_vars = sqrt(feature.cols);
	_params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 50, 0.1);
*/
	double t = double(getTickCount());
	cout << "  Training random forest regression model ...";

	Mat varType = Mat::ones(feature.cols+1,1,CV_8UC1) * CV_VAR_NUMERICAL;

	_random_tree.train(feature, CV_ROW_SAMPLE, labels, Mat(), Mat(), varType, Mat(), _params);

	t = (getTickCount()-t)/getTickFrequency();
	cout << " time to train:" << t << " secs." << endl;
	_isTrained = true;
    return;
}

Mat LcRandomTreesR::predict(Mat & feature)
{
    int n = feature.rows;
	Mat res = Mat::zeros( n, 1, CV_32F);
	for(int i = 0; i < n; i++)
	{
		res.at<float>(i,0) =  _random_tree.predict(feature.row(i), Mat());
	}

	return res;
}

Mat LcRandomTreesR::predict_prob(Mat & feature)
{
    return predict(feature);
}

void LcRandomTreesC::save(string filename)
{
    cout << "\nClassifier: Saving " << filename << endl;
	_random_tree.save( filename.c_str());
}

void LcRandomTreesC::load(string filename)
{
    cout << "Classifier: Loading " << filename << endl;
    _random_tree.load( filename.c_str());
}

void LcRandomTreesC::train(Mat &feature, Mat &labels, string dataFile, string modelFile)
{
	_params.max_depth				= 10;
	_params.regression_accuracy		= 0.00f;
	_params.min_sample_count		= 10;
/*
	_params.calc_var_importance 	= true;
	_params.nactive_vars = sqrt(feature.cols);
	_params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 50, 0.1);
*/
	double t = double(getTickCount());
	cout << "  Training random forest classification model ...";

	Mat varType = Mat::ones(feature.cols+1,1,CV_8UC1) * CV_VAR_NUMERICAL;
	varType.at<uchar>(feature.cols,0) = CV_VAR_CATEGORICAL;
	_random_tree.train(feature, CV_ROW_SAMPLE, labels, Mat(), Mat(), varType, Mat(), _params);

	t = (getTickCount()-t)/getTickFrequency();
	cout << " time to train:" << t << " secs." << endl;

    return;
}

Mat LcRandomTreesC::predict(Mat & feature)
{
    int n = feature.rows;
	Mat res = Mat::zeros( n, 1, CV_32F);
	for(int i = 0; i < n; i++)
	{
		Mat data(1, feature.cols, feature.type());
		feature.row(i).copyTo(data);
		res.at<float>(i,0) =  _random_tree.predict(data, Mat());
	}

	return res;
}

//only support binary classifier
Mat LcRandomTreesC::predict_prob(Mat & feature)
{
    int n = feature.rows;
	Mat res = Mat::zeros( n, 1, CV_32F);
	for(int i = 0; i < n; i++)
	{
		Mat data(1, feature.cols, feature.type());
		feature.row(i).copyTo(data);
		res.at<float>(i,0) =  1.0 - _random_tree.predict_prob(data, Mat());
	}

	return res;
}

