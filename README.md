# MatrixMethods
Matrix methods C++ library for machine learning

### Example Usage
    #include <MatrixMethods>
    
    int main() {
      int batch_size = 5000;
    
      mm::MatrixDataLoader train_data(
        "../data/mnist-fashion/X_train.csv", 
        "../data/mnist-fashion/y_train.csv",
        784, 10, batch_size
      );

      double lambda = 1e-3
    
      mm::ANN model(
        {
          new mm::DenseL2(28 * 28, 128, lambda),
          new mm::ReLU(),
          new mm::DenseL2(128, 128, lambda),
          new mm::ReLU(),
          new mm::DenseL2(128, 10, lambda),
        },
        new mm::SoftmaxCategoricalCrossentropy()
      );
    
      double learning_rate = 1e-3;
      
      for (int batch_num = 0; batch_num < 1000; batch_num++) {
        auto [X, Y] = train_data.get_batch();
        for (int epoch = 0; epoch < 10; epoch++) {
          for (int i = 0; i < batch_size; i++) {
            model.predict(X.row(i));
            model.update(Y.row(i), learning_rate);
          }
        }
      }
    }
    
### Features
- Linear Regression (L1 and L2 regularization)
- Support Vector Machine
- Kernel Regression (L2 regularization and Support Vector Machine)
- Neural Networks (Convolutional and Dense layers)
- Data Loader 
    
