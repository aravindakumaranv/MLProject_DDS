import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go

class DataImputer:
    def __init__(self, column_to_value):
        self.column_to_value = column_to_value
    
    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:
        for column in self.column_to_value.keys():
            data[column] = data[column].fillna(self.column_to_value[column])
            return data

class ArrivalDateTransformer:
    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:
        # Converting arrival_date_month values from string to integer
        data['arrival_date_month'].replace(
            ['July', 'August', 'September', 'October', 'November', 'December', 'January', 'February', 'March', 'April', 'May', 'June'],
            [7,8,9,10,11,12,1,2,3,4,5,6],
            inplace=True
        )

        data.rename(
            columns = {
                'arrival_date_year': 'year', 
                'arrival_date_month': 'month',
                'arrival_date_day_of_month': 'day'
            }, 
            inplace=True
        )

        # Creating new arrival_date column of type datetime
        data.insert(3,'arrival_date',pd.to_datetime(data[['year', 'month', 'day']]))
        return data

class ColumnRemover:
    def __init__(self, columns):
        self.columns = columns

    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:
        data.drop(self.columns, axis = 1, inplace=True)
        return data

class RoomTypeTransformer:
    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:
        data['reserved_assigned_match'] = np.where(data['reserved_room_type'] == data['assigned_room_type'], 0, 1)
        data.drop(['reserved_room_type','assigned_room_type'], axis=1, inplace=True)
        return data

class MealTypeTransformer:
    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:
        data['meal'].replace(['Undefined', 'SC', 'BB', 'HB', 'FB'], [0, 0, 0.33, 0.67, 1], inplace=True)
        return data

class UncleanDataPointsRemover:
    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:
        data.drop(
            data[
                (data['adults']==0) &
                (data['children']==0) &
                (data['babies']==0) &
                (data['stays_in_weekend_nights']==0) & 
                (data['stays_in_week_nights']==0) &
                (data['is_canceled']==0)
            ].index,
            inplace=True
        )

        data.drop(
            data[
                (data["adr"]==0) &  
                (data["is_canceled"]==0)  &
                (data["market_segment"] != "Complementary") &
                (data["market_segment"] != "Corporate") &
                (data["market_segment"] != "Aviation") 
            ].index, 
            inplace = True
        )        

        data.drop(
            data[
                (data["stays_in_week_nights"]==0) &
                (data["stays_in_weekend_nights"]==0) &
                (data["arrival_date"] == data["reservation_status_date"]) &
                (data["reservation_status"]!="Check-Out")
            ].index,
            inplace=True
        )

        return data

class CancellationsDaysInserter:
    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:
        # Creating cancellation_days column which represents how many days in advance the guest cancels 
        data['reservation_status_date'] = pd.to_datetime(data['reservation_status_date'], format = '%Y-%m-%d')
        data['cancellation_days'] = data['arrival_date'] - data['reservation_status_date']
        data['cancellation_days'] = data['cancellation_days'].dt.days
        return data

def plot_confusion_matrix(true_values, predicted_values):
    
    confusion_matrix_values = confusion_matrix(true_values, predicted_values)

    heading = [
        'Confusion Matrix',
        'True Positive',
        'True Negative'
    ]
    values = [
        ['Predicted Positive','Predicted Negative'],
        [confusion_matrix_values[0][0],confusion_matrix_values[1][0]],
        [confusion_matrix_values[0][1],confusion_matrix_values[1][1]]
    ]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=heading,
                    fill_color='royalblue',
                    font=dict(color='white', size=12),
                ),
                cells=dict(
                    values=values,
                    fill=dict(color=['royalblue', 'paleturquoise']),
                    font=dict(color=['white','black'], size=12),
                )
            )
        ]
    )
    fig.update_layout(width=500, height=400)
    fig.show()
    
def generate_classification_report(true_values, predicted_values, target_names):
    return classification_report(
        true_values, predicted_values, target_names
    )