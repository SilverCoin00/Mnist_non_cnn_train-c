#include <stdio.h>
#include "D:\Data\code_doc\AI_model_building\Multi-layer_perceptron\Core.h"

int main() {
    /*trans_images_to_csv("D:\\Data\\mnist_non_cnn_training\\archive\\trainingSet\\trainingSet", 1, 10, 
                            "D:\\Data\\mnist_non_cnn_training\\train_data.csv", ',');*/
            
    Data_Frame* df;
    df = read_csv("D:\\Data\\mnist_non_cnn_training\\test_data.csv", 4000, ",");
    Dataset_2* ds = trans_dframe_to_dset2(df, "y");
    free_data_frame(df);

    Standard_scaler* scaler = (Standard_scaler*) load_scaler("D:\\Data\\mnist_non_cnn_training\\mnist_scaler.txt", "Standard_scaler");
    scaler_transform(ds->x, NULL, ds->samples, ds->features, scaler, "Standard_scaler");
    free_scaler(scaler, "Standard_scaler");

    MLP* model = load_model("D:\\Data\\mnist_non_cnn_training\\model_2.txt");
    model_compile(model, NULL, "categorical_crossentropy", "categorical_accuracy");
    model_evaluate(model, ds);

    free_mlp_model(model);
    free_dataset2(ds);
}