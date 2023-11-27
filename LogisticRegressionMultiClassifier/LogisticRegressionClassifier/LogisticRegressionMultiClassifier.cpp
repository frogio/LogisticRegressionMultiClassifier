#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#pragma warning(disable:4996)

#define CLASS_SETOSA					0
#define CLASS_VERSICOLOR				1
#define CLASS_VIRGINICA					2

#define MAX_DATA						150							// ��ü ������ ����, M
#define ALPHA							0.01						// �н���
#define EPOCH							10000						// �н� Ƚ��

struct Model {
	double w0;
	double w1;
	double w2;
};

struct Target { 
	double sepalLen;
	double petalLen;
	int _class[3];													// Ŭ���� ����, OneHot Encoding
};

Target * LoadData();												// iris dataset�� �޾ƿ´�.
void Training(struct Target * target, struct Model * model);
double * SoftMax(struct Model* model, double sepalLen, double petalLen);
int Predict(struct Model* model, double sepalLen, double  petalLen);

void main() {

	const char* className[3] = {"Virginica" ,"Versicolor", "Setosa" };

	struct Target * target = LoadData();

	struct Model model[3] = {{ 1, 1, 1 },										// Setasa ���� �з� ��
								{ 1, 1, 1 },									// Versicolor �����з� ��
								{ 1, 1, 1 }};									// Virginica �����з� ��
								// �н����� ���� �ʱ� ��

	printf("Loaded Data...\n\n");
	for (int i = 0; i < MAX_DATA; i++){
		printf("sepalLength, petalLength, OneHot Encoded class: %lf, %lf ", target[i].sepalLen, target[i].petalLen);
	
		for (int k = 0; k < 3; k++)
			printf("%d ", target[i]._class[k]);
		
		printf("\n");
	
	}

	for (int i = 0; i < EPOCH; i++)
		Training(target, model);

	printf("Training Result : \n");
	
	for(int i = 0; i < 3; i++)
		printf("y = %lf * x2 + %lf * x1 + %lf\n", model[i].w2, model[i].w1, model[i].w0);
	// �н��� �з��𵨵� ���

	double sepalLen, petalLen;
	
	while (1) {
		printf("\nEnter sepalLength, petelLength (exit -1, -1): ");
		scanf("%lf,%lf", &sepalLen, &petalLen);
		
		printf("Predict Result : %s", className[Predict(model, sepalLen, petalLen)]);
		
		if (sepalLen == -1)
			break;

	}

	free(target);
}

int Predict(struct Model* model, double sepalLen, double  petalLen) {

	double * prob = SoftMax(model, sepalLen, petalLen);

	double max = prob[0];
	int result = 0;

	for (int i = 1; i < 3; i++) {
		if (max < prob[i]){
			max = prob[i];
			result = i;
		}
	}										// softmax ��������� �ִ밪�� ������ ��,


	return result;							// Ȯ���� ���� ū ���� ������.
}

double* SoftMax(struct Model* model, double sepalLen, double petalLen) {

	double u[3] = { 0.f, },						// ȸ�͸� �����
		exp_u[3] = { 0.f, };					// �ڿ��α� �����


	//double u0 = model[0].w2 * sepelLen + model[0].w1 * sepelWidth + model[0].w0 * 1;
	//double u1 = model[1].w2 * sepelLen + model[1].w1 * sepelWidth + model[1].w0 * 1;
	//double u2 = model[2].w2 * sepelLen + model[2].w1 * sepelWidth + model[2].w0 * 1;

	for (int i = 0; i < 3; i++)			// ȸ�͸� ����
		u[i] = model[i].w2 * sepalLen + model[i].w1 * petalLen + model[i].w0 * 1;	// ��� ���, 3���� ȸ�͸𵨿� �Է°��� ����ִ´�.				
		//printf("exp_u : %lf\n", exp_u[i]);	

	double max = u[0];

	for (int i = 1; i < 3; i++) {
		if (u[i] > max)
			max = u[i];	
	}

	for(int i = 0; i < 3; i++)
		exp_u[i] = exp(u[i] - max);				// �ڿ���� �� ���
	// �����÷ο� ������ ���� ȸ�͸��� ������ ���� ū ���� ���� ���ش�.

	double * result = (double *)malloc(sizeof(double) * 3);
	
	if(result){

		for (int i = 0; i < 3; i++)				// softmax �����
			result[i] = exp_u[i] / (exp_u[0] + exp_u[1] + exp_u[2]);
	
	}

	return result;								// SoftMax ��� ���� ����

}
void Training(struct Target* target, struct Model* model) {									// ����ϰ����� �̿��� Training

	Model diff_vec[3] = { {0.f, }, };

	for (int i = 0; i < MAX_DATA; i++) {

		double* predVal = SoftMax(model, target[i].sepalLen, target[i].petalLen);					// �� 3���� ���� �Է°� ���� (��� * ����)
		double error[3] = { 0.f, };																	// ������ ���

		for (int j = 0; j < 3; j++)																	// ���Ͱ� ������ ���Ͽ� ���� ���
			error[j] = predVal[j] - target[i]._class[j];
			// ������ OneHotVector�� Predict���� ���� ���, => (1, 0, 0), (0, 1, 0), (0, 0, 1)		// ���Ͱ� ����


		for (int k = 0; k < 3; k++) {

			diff_vec[k].w0 += error[k];
			diff_vec[k].w1 += error[k] * target[i].sepalLen;
			diff_vec[k].w2 += error[k] * target[i].petalLen;
		
		}



		free(predVal);																		// �ӽ� Predict ���� Free �Ͽ� �Ҵ� ����

	}

	for (int k = 0; k < 3; k++){

		diff_vec[k].w0 /= MAX_DATA;
		diff_vec[k].w1 /= MAX_DATA;
		diff_vec[k].w2 /= MAX_DATA;
		// ������� �ս��Լ� �̺а� ���

		model[k].w0 -= diff_vec[k].w0 * ALPHA;
		model[k].w1 -= diff_vec[k].w1 * ALPHA;
		model[k].w2 -= diff_vec[k].w2 * ALPHA;
	
	}
	// ��� �ϰ����� ���� ����ġ ����
	// �н����� (���� �̵�����) ���� �� ���� ����ġ ����

}

Target * LoadData() {
	
	char buf[200] = { 0, };
	struct Target * target;

	target = (struct Target *)malloc(sizeof(Target) * MAX_DATA);

	FILE* fp = fopen("iris data.csv", "rt");
	
	fgets(buf, sizeof(buf), fp);								// csv ������ ���κ��� ���� �о�´�.

	int tIdx = 0;
	for(int i = 0; i < MAX_DATA; i++){

		fgets(buf, sizeof(buf), fp);
		
		char * data = buf;

		target[tIdx].sepalLen = atof(data);							// sepalLength ������ ����

		for (int i = 0; i < 2; i++) {
			data = strchr(buf, ',');
			*data = ' ';
		}
		data++;

		target[tIdx].petalLen = atof(data);							// petalLength ������ ����
		
		data = strchr(buf, ',');
		*data = ' ';
		data++;

		target[tIdx]._class[0] = 0;
		target[tIdx]._class[1] = 0;
		target[tIdx]._class[2] = 0;

		target[tIdx]._class[
			((strstr(data, "Setosa") != NULL) ? CLASS_SETOSA :
				(strstr(data, "Versicolor") != NULL) ?
				CLASS_VERSICOLOR : CLASS_VIRGINICA)] = 1;			// Ŭ���� OneHotEncoding
		
		tIdx++;

	}
	fclose(fp);

	return target;

}