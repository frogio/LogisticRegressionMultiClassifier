#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#pragma warning(disable:4996)

#define CLASS_SETOSA					0
#define CLASS_VERSICOLOR				1
#define CLASS_VIRGINICA					2

#define MAX_DATA						150							// 전체 데이터 개수, M
#define ALPHA							0.01						// 학습률
#define EPOCH							10000						// 학습 횟수

struct Model {
	double w0;
	double w1;
	double w2;
};

struct Target { 
	double sepalLen;
	double petalLen;
	int _class[3];													// 클래스 벡터, OneHot Encoding
};

Target * LoadData();												// iris dataset을 받아온다.
void Training(struct Target * target, struct Model * model);
double * SoftMax(struct Model* model, double sepalLen, double petalLen);
int Predict(struct Model* model, double sepalLen, double  petalLen);

void main() {

	const char* className[3] = {"Virginica" ,"Versicolor", "Setosa" };

	struct Target * target = LoadData();

	struct Model model[3] = {{ 1, 1, 1 },										// Setasa 이진 분류 모델
								{ 1, 1, 1 },									// Versicolor 이진분류 모델
								{ 1, 1, 1 }};									// Virginica 이진분류 모델
								// 학습되지 않은 초기 모델

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
	// 학습된 분류모델들 출력

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
	}										// softmax 결과값에서 최대값을 선택한 후,


	return result;							// 확률이 가장 큰 값이 예측값.
}

double* SoftMax(struct Model* model, double sepalLen, double petalLen) {

	double u[3] = { 0.f, },						// 회귀모델 결과값
		exp_u[3] = { 0.f, };					// 자연로그 결과값


	//double u0 = model[0].w2 * sepelLen + model[0].w1 * sepelWidth + model[0].w0 * 1;
	//double u1 = model[1].w2 * sepelLen + model[1].w1 * sepelWidth + model[1].w0 * 1;
	//double u2 = model[2].w2 * sepelLen + model[2].w1 * sepelWidth + model[2].w0 * 1;

	for (int i = 0; i < 3; i++)			// 회귀모델 적용
		u[i] = model[i].w2 * sepalLen + model[i].w1 * petalLen + model[i].w0 * 1;	// 행렬 계산, 3개의 회귀모델에 입력값을 집어넣는다.				
		//printf("exp_u : %lf\n", exp_u[i]);	

	double max = u[0];

	for (int i = 1; i < 3; i++) {
		if (u[i] > max)
			max = u[i];	
	}

	for(int i = 0; i < 3; i++)
		exp_u[i] = exp(u[i] - max);				// 자연상수 값 계산
	// 오버플로우 방지를 위해 회귀모델의 값에서 가장 큰 값을 구해 빼준다.

	double * result = (double *)malloc(sizeof(double) * 3);
	
	if(result){

		for (int i = 0; i < 3; i++)				// softmax 결과값
			result[i] = exp_u[i] / (exp_u[0] + exp_u[1] + exp_u[2]);
	
	}

	return result;								// SoftMax 결과 벡터 리턴

}
void Training(struct Target* target, struct Model* model) {									// 경사하강법을 이용한 Training

	Model diff_vec[3] = { {0.f, }, };

	for (int i = 0; i < MAX_DATA; i++) {

		double* predVal = SoftMax(model, target[i].sepalLen, target[i].petalLen);					// 모델 3개에 대해 입력값 적용 (행렬 * 벡터)
		double error[3] = { 0.f, };																	// 에러율 계산

		for (int j = 0; j < 3; j++)																	// 벡터간 뺄셈을 통하여 에러 계산
			error[j] = predVal[j] - target[i]._class[j];
			// 예측값 OneHotVector와 Predict간의 에러 계산, => (1, 0, 0), (0, 1, 0), (0, 0, 1)		// 벡터간 뺄셈


		for (int k = 0; k < 3; k++) {

			diff_vec[k].w0 += error[k];
			diff_vec[k].w1 += error[k] * target[i].sepalLen;
			diff_vec[k].w2 += error[k] * target[i].petalLen;
		
		}



		free(predVal);																		// 임시 Predict 값은 Free 하여 할당 해제

	}

	for (int k = 0; k < 3; k++){

		diff_vec[k].w0 /= MAX_DATA;
		diff_vec[k].w1 /= MAX_DATA;
		diff_vec[k].w2 /= MAX_DATA;
		// 여기까지 손실함수 미분값 계산

		model[k].w0 -= diff_vec[k].w0 * ALPHA;
		model[k].w1 -= diff_vec[k].w1 * ALPHA;
		model[k].w2 -= diff_vec[k].w2 * ALPHA;
	
	}
	// 경사 하강법을 통한 가중치 조정
	// 학습률을 (벡터 이동방향) 곱한 후 더해 가중치 조정

}

Target * LoadData() {
	
	char buf[200] = { 0, };
	struct Target * target;

	target = (struct Target *)malloc(sizeof(Target) * MAX_DATA);

	FILE* fp = fopen("iris data.csv", "rt");
	
	fgets(buf, sizeof(buf), fp);								// csv 파일의 헤드부분을 먼저 읽어온다.

	int tIdx = 0;
	for(int i = 0; i < MAX_DATA; i++){

		fgets(buf, sizeof(buf), fp);
		
		char * data = buf;

		target[tIdx].sepalLen = atof(data);							// sepalLength 데이터 추출

		for (int i = 0; i < 2; i++) {
			data = strchr(buf, ',');
			*data = ' ';
		}
		data++;

		target[tIdx].petalLen = atof(data);							// petalLength 데이터 추출
		
		data = strchr(buf, ',');
		*data = ' ';
		data++;

		target[tIdx]._class[0] = 0;
		target[tIdx]._class[1] = 0;
		target[tIdx]._class[2] = 0;

		target[tIdx]._class[
			((strstr(data, "Setosa") != NULL) ? CLASS_SETOSA :
				(strstr(data, "Versicolor") != NULL) ?
				CLASS_VERSICOLOR : CLASS_VIRGINICA)] = 1;			// 클래스 OneHotEncoding
		
		tIdx++;

	}
	fclose(fp);

	return target;

}