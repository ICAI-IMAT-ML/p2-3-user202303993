# Import here whatever you may need
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class LinearRegressor:
    """
    Linear Regression model that can perform both simple and multiple linear regression.

    Attributes:
        coefficients (np.ndarray): Coefficients of the independent variables in the regression model.
        intercept (float): Intercept of the regression model.
    """

    def __init__(self):
        """Initializes the LinearRegressor model with default coefficient and intercept values."""
        self.coefficients = None #coeficiente de pendiente de la recta 
        self.intercept = None #valor de y cuando x es cero  

    def fit_simple(self, X, y): #encontramos la mejor linea recta que relacione  X(Var indep) con Y 
        """
        Fit the model using simple linear regression (one independent variable).

        This method calculates the coefficients for a linear relationship between
        a single predictor variable X and a response variable y.

        Args:
            X (np.ndarray): Independent variable data (1D array).
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if np.ndim(X) > 1:
            X = X.reshape(1, -1)   
         
        # TODO: Train linear regression model with only one coefficient
         #utlizamos las formulas que obtuvimos en clase de estimacion de parámetros (derivando e igualando a 0)
        X_mean = np.mean(X)
        y_mean = np.mean(y)
        
        num = np.sum((X - X_mean) * (y - y_mean))  #Sumatoria del producto de desviaciones de x e y 
        denom = np.sum((X - X_mean) ** 2)  # Sumatoria del cuadrado de desviaciones de X
        m = num / denom
        self.coefficients = m
        
        b = y_mean - m * X_mean
        self.intercept = b  
        
         
    # This part of the model you will only need for the last part of the notebook
    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array where each column is a variable).
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        # TODO: Train linear regression model with multiple coefficients
        import numpy as np

        #Asegurar que X es un array 2D
        """
        X debe ser una matriz en 2 dimensiones por 
        - Filas son las observaciones 
        - Columnas cada uno de los regresores 
        """
        X = np.asarray(X) #covertimos en arrey de numpy por si acaso 
        if X.ndim == 1: #si la dimension de x es 1 signifca solo hay una var indep(regresor) que puede tener varias obsevaciones
            X = X.reshape(-1, 1) #conviertes vectro en fila en matriz con una sola columna 
        
        #Añadimos una columna de unos para el intercepto, ya que la primera fila de nuetsro vector de w tendrá el intercepto b (b,w1,w2,...,wn) 
        X_b = np.c_[np.ones(X.shape[0]), X]  #np.c_ concatena columnas

        #Aplicamos la fórmula de regresión múltiple (OJO  el signo  @ multiplica matrices entre ellas)
        beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y  #Formula derivada de la funcion de perdida 1)Expando matrices 2) Derivo respecto w 3)Despeko w (X^T X)^(-1) X^T y

        #Asignamos coeficientes e intercepto
        self.intercept = beta[0]
        self.coefficients = beta[1:]

    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """
        if self.coefficients is None or self.intercept is None: #estos atributos los hemos sacado con las 3 funciones anteriores y esta funcion nos obliga a que no sean None
            raise ValueError("Model is not yet fitted")

        if np.ndim(X) == 1:
            # TODO: Predict when X is only one variable
            predictions = self.coefficients * X + self.intercept
        else:
            # TODO: Predict when X is more than one variable
            predictions = X @ self.coefficients + self.intercept  
        return predictions


def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """
    # R^2 Score
    # TODO: Calculate R^2
    RSS = np.sum((y_true - np.mean(y_true)) ** 2) #RSS Varianza respecto a la media  
    TSS = np.sum((y_true - y_pred) ** 2) #TSS Varianza respecto a la curva  
    r_squared = 1 - (TSS / RSS)

    # Root Mean Squared Error
    # TODO: Calculate RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Mean Absolute Error
    # TODO: Calculate MAE
    mae = np.mean(np.abs(y_true - y_pred))

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}


# ### Scikit-Learn comparison


def sklearn_comparison(x, y, linreg):
    """Compares a custom linear regression model with scikit-learn's LinearRegression.

    Args:
        x (numpy.ndarray): The input feature data (1D array).
        y (numpy.ndarray): The target values (1D array).
        linreg (object): An instance of a custom linear regression model. Must have
            attributes `coefficients` and `intercept`.

    Returns:
        dict: A dictionary containing the coefficients and intercepts of both the
            custom model and the scikit-learn model. Keys are:
            - "custom_coefficient": Coefficient of the custom model.
            - "custom_intercept": Intercept of the custom model.
            - "sklearn_coefficient": Coefficient of the scikit-learn model.
            - "sklearn_intercept": Intercept of the scikit-learn model.
    """
    ### Compare your model with sklearn linear regression model
    # TODO : Import Linear regression from sklearn
    from sklearn.linear_model import LinearRegression #importamos

    # Assuming your data is stored in x and y
    # TODO : Reshape x to be a 2D array, as scikit-learn expects 2D inputs for the features
    x_reshaped = x.reshape(-1, 1)#hacemos el reshape que nos piden
 
    # Create and train the scikit-learn model
    # TODO : Train the LinearRegression model
    sklearn_model = LinearRegression() #llamamos a LinearRegression
    sklearn_model.fit(x_reshaped, y)

    # Now, you can compare coefficients and intercepts between your model and scikit-learn's model
    print("Custom Model Coefficient:", linreg.coefficients)
    print("Custom Model Intercept:", linreg.intercept)
    print("Scikit-Learn Coefficient:", sklearn_model.coef_[0])
    print("Scikit-Learn Intercept:", sklearn_model.intercept_)
    return {
        "custom_coefficient": linreg.coefficients,
        "custom_intercept": linreg.intercept,
        "sklearn_coefficient": sklearn_model.coef_[0],
        "sklearn_intercept": sklearn_model.intercept_,
    }

def anscombe_quartet():
    """Loads Anscombe's quartet, fits custom linear regression models, and evaluates performance.

    Returns:
        tuple: A tuple containing:
            - anscombe (pandas.DataFrame): The Anscombe's quartet dataset.
            - datasets (list): A list of unique dataset identifiers in Anscombe's quartet.
            - models (dict): A dictionary where keys are dataset identifiers and values
              are the fitted custom linear regression models.
            - results (dict): A dictionary containing evaluation metrics (R2, RMSE, MAE)
              for each dataset.
    """
    # Load Anscombe's quartet
    # These four datasets are the same as in slide 19 of chapter 02-03: Linear and logistic regression
    anscombe = sns.load_dataset("anscombe")

    # Anscombe's quartet consists of four datasets
    # TODO: Construct an array that contains, for each entry, the identifier of each dataset
    datasets = anscombe['dataset'].unique() #simplemnte vemos al hacer print en la columna de datasets que hay 4 tipos I,II,III,IV

    models = {}
    results = {"R2": [], "RMSE": [], "MAE": []}
    for dataset in datasets:

        # Filter the data for the current dataset
        # TODO
        data = anscombe[anscombe["dataset"] == dataset]

        # Create a linear regression model
        # TODO
        model = LinearRegressor() #OJO ESO NO ES el model de sk sino el modelo que hemos creado nosotros, nuestro obejto con las funciones programadas era LinearRegressor

        # Fit the model
        # TODO 
        #obtenemos la X y la y del dataset
        X = data["x"].values   # Predictor, make it 1D for your custom model
        y = data["y"].values  # Response
        model.fit_simple(X, y)

        # Create predictions for dataset
        # TODO
        y_pred =model.predict(X) #usamos la funcion programada predict 

        # Store the model for later use
        models[dataset] = model

        # Print coefficients for each dataset
        print(
            f"Dataset {dataset}: Coefficient: {model.coefficients}, Intercept: {model.intercept}"
        )

        evaluation_metrics = evaluate_regression(y, y_pred)

        # Print evaluation metrics for each dataset
        print(
            f"R2: {evaluation_metrics['R2']}, RMSE: {evaluation_metrics['RMSE']}, MAE: {evaluation_metrics['MAE']}"
        )
        results["R2"].append(evaluation_metrics["R2"])
        results["RMSE"].append(evaluation_metrics["RMSE"])
        results["MAE"].append(evaluation_metrics["MAE"])

    return anscombe, datasets, models, results


