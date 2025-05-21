#include <stdio.h>
#include "D:\Data\code_doc\AI_model_building\Multi-layer_perceptron\Core.h"
#include "D:\Std_gcc_&_g++_plus\stb_image_supporting.c"

int main() {
    /*trans_images_to_csv("D:\\Data\\mnist_non_cnn_training\\archive\\trainingSet\\trainingSet", 1, 10, 
                            "D:\\Data\\mnist_non_cnn_training\\train_data.csv", ',');*/
            
    Data_Frame* df;
    df = read_csv("D:\\Data\\mnist_non_cnn_training\\train_data.csv", 4000, ",");
    Dataset_2* train_ds = trans_dframe_to_dset2(df, "y");
    free_data_frame(df);
    df = read_csv("D:\\Data\\mnist_non_cnn_training\\val_data.csv", 4000, ",");
    Dataset_2* val_ds = trans_dframe_to_dset2(df, "y");
    free_data_frame(df);

    Standard_scaler* scaler = (Standard_scaler*) load_scaler("D:\\Data\\mnist_non_cnn_training\\mnist_scaler.txt", "Standard_scaler");
    scaler_transform(train_ds->x, NULL, train_ds->samples, train_ds->features, scaler, "Standard_scaler");
    scaler_transform(val_ds->x, NULL, val_ds->samples, val_ds->features, scaler, "Standard_scaler");
    //scaler_save("D:\\Data\\mnist_non_cnn_training\\mnist_scaler.txt", scaler, "Standard_scaler");
    free_scaler(scaler, "Standard_scaler");

    /*MLP* model = new_sequential_model((int[]) {train_ds->samples, train_ds->features}, 
                            (Keras_layer*[]) {&(Keras_layer) {&(Dense) {128, "relu"}, NULL}, 
                                            &(Keras_layer) {NULL, &(Dropout) {0.2}}, 
                                            &(Keras_layer) {&(Dense) {32, "relu"}, NULL}, 
                                            &(Keras_layer) {NULL, &(Dropout) {0.2}}, 
                                            &(Keras_layer) {&(Dense) {10, "softmax"}, NULL}, NULL});*/
    MLP* model = load_model("D:\\Data\\mnist_non_cnn_training\\mnist_model.txt");
    Optimizer* opt = new_optimizer("SGD", 0.01, 0.9, 1, Nan, Nan, Nan, Nan, Nan);
    //Optimizer* opt = new_optimizer("Adam", 0.008, Nan, Nan, 0.9, 0.999, Nan, 1e-8, Nan);
    Early_Stopping* estop = new_earlystopping("val_loss", 1e-6, 3, Nan, 1);
    model_compile(model, opt, "categorical_crossentropy", "categorical_accuracy");
    model->val_data = val_ds;
    model_fit(model, train_ds, 10, 64, -1, estop);
    model_save("D:\\Data\\mnist_non_cnn_training\\mnist_model.txt", model);

    free_mlp_model(model);
    free(estop);
    free_dataset2(train_ds);
    free_dataset2(val_ds);
}