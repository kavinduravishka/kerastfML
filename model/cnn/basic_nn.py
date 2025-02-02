from tensorflow import keras

class basic_nn:
    def __init__(self,labelcount,input_shape):
        self.labelcount=labelcount
        self.input_shape=input_shape
        self.model=None
    
    def compilemodel(self,optimizer,learning_rate,loss,metrics):
        self.model.compile(optimizer=optimizer(learning_rate=learning_rate),loss=loss,metrics=metrics)
        
    def fit(self,train_batches,valid_batches,epochs=20):
        fit_history = self.model.fit(x=train_batches.iterset,
            steps_per_epoch=train_batches.length,
            validation_data=valid_batches.iterset,
            validation_steps=valid_batches.length,
            epochs=epochs,
            verbose=2
            )
        return fit_history
        
    def predict(self,test_batches):
        self.predictions = self.model.predict(x=test_batches.iterset, steps=test_batches.length, verbose=0)
        return self
    
    def getresult(self):
        return self.predictions
    
    def summary(self):
        self.model.summary()
        
    def serialize(self,name):
        self.model.save(name)
    
    def deserialize(self,name):
        self.model = keras.models.load_model(name)