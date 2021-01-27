class basic_cnn:
    def __init__(self,labelcount,input_shape):
        self.labelcount=labelcount
        self.input_shape=input_shape
        self.model=None
    
    def compilemodel(self,optimizer,learning_rate,loss,metrices):
        self.model.compile(optimizer=optimizer(learning_rate=learning_rate),loss=loss,metrices=metrices)
        
    def fit(self,train_batches,valid_batches,epochs=10):
        self.model.fit(x=train_batches,
            steps_per_epoch=len(train_batches),
            validation_data=valid_batches,
            validation_steps=len(valid_batches),
            epochs=epochs,
            verbose=2
            )
        
    def predict(self,test_batches):
        self.predictions = self.model.predict(x=test_batches, steps=len(test_batches), verbose=0)
        return self
    
    def getresult(self):
        return self.predictions
    
    def summary(self):
        self.model.summary()