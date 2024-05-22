from flask import Blueprint, render_template, request, flash, jsonify, redirect
from flask_login import login_required, current_user
from .models import Note
from .models import Event
from . import db
import json
import pandas as pd
import plotly
import plotly.express as px
from sqlalchemy.sql import func
import plotly.graph_objs as go
import numpy as np  # Import NumPy for statistical calculations
from sklearn.linear_model import LogisticRegression  # Import Logistic Regression model
from sklearn.svm import SVR  # Import Support Vector Regressor
from sklearn.tree import DecisionTreeRegressor  # Import Decision Tree Regressor
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

views = Blueprint('views', __name__)

from datetime import datetime, timedelta

@views.route('/prediction')
@login_required
def prediction():

    # Query the capacity and timestamp data for the current user
    events = Event.query.filter_by(user_id=current_user.id).order_by(Event.timestamp).all()

    if not events:
        flash('No events data available for prediction', category='error')
        return render_template('prediction.html', predictions=None)

    # Prepare data for the regression models
    timestamps = [event.timestamp for event in events]
    capacities = [event.capacity for event in events]

    # Convert timestamps to numeric values (e.g., seconds since the first timestamp)
    time_deltas = [(timestamp - timestamps[0]).total_seconds() for timestamp in timestamps]

    # Create a DataFrame
    df = pd.DataFrame({
        'time_delta': time_deltas,
        'capacity': capacities
    })

    # Define the models
    models = {
        'Prediction 1': LogisticRegression(),
        'Prediction 2': SVR(),
        'Prediction 3': DecisionTreeRegressor()
    }

    future_time_delta = (timestamps[-1] - timestamps[0]).total_seconds() + 7200
    predictions = {}

    # Train each model and make predictions
    for model_name, model in models.items():
        model.fit(df[['time_delta']], df['capacity'])
        predicted_capacity = model.predict([[future_time_delta]])[0]
        predictions[model_name] = predicted_capacity


    # Query the event data for the current user
    events = Event.query.filter_by(user_id=current_user.id).order_by(Event.timestamp).all()

    if not events:
        flash('No events data available for prediction', category='error')
        return render_template('prediction.html', predictions=None)

    # Prepare data for the regression models
    timestamps = [event.timestamp for event in events]
    event_types = [event.event_type for event in events]

    # Convert timestamps to numeric values (e.g., seconds since the first timestamp)
    time_deltas = [(timestamp - timestamps[0]).total_seconds() for timestamp in timestamps]

    # Calculate cumulative not_running duration
    cumulative_not_running = []
    total_not_running = 0
    for i in range(len(events)):
        if event_types[i] == '0':
            if i < len(events) - 1:
                duration = (timestamps[i + 1] - timestamps[i]).total_seconds()
            else:
                duration = 0
            total_not_running += duration
        cumulative_not_running.append(total_not_running)

    # Create a DataFrame
    df = pd.DataFrame({
        'time_delta': time_deltas,
        'cumulative_not_running': cumulative_not_running
    })

    # Define the models
    models = [
        LinearRegression(),
        SVR(),
        DecisionTreeRegressor()
    ]

    future_time_delta = time_deltas[-1] + 7200
    predictions2 = []

    # Train each model and make predictions
    for model in models:
        model.fit(df[['time_delta']], df['cumulative_not_running'])
        predicted_cumulative_downtime = model.predict([[future_time_delta]])[0]
        current_cumulative_downtime = df['cumulative_not_running'].iloc[-1]
        predicted_downtime_next_2_hours = predicted_cumulative_downtime - current_cumulative_downtime
        predictions2.append(predicted_downtime_next_2_hours)

    # Calculate the average prediction
    final_prediction = sum(predictions2) / len(predictions2)

    # Query the event data for the current user
    events = Event.query.filter_by(user_id=current_user.id).order_by(Event.timestamp).all()

    if not events:
        flash('No events data available for prediction', category='error')
        return render_template('prediction.html', predictions=None)

    # Prepare data for the regression models
    timestamps = [event.timestamp for event in events]
    event_types = [event.event_type for event in events]

    # Convert timestamps to numeric values (e.g., seconds since the first timestamp)
    time_deltas = [(timestamp - timestamps[0]).total_seconds() for timestamp in timestamps]

    # Calculate cumulative not_running duration
    cumulative_not_running = []
    total_not_running = 0
    for i in range(len(events)):
        if event_types[i] == '1':
            if i < len(events) - 1:
                duration = (timestamps[i + 1] - timestamps[i]).total_seconds()
            else:
                duration = 0
            total_not_running += duration
        cumulative_not_running.append(total_not_running)

    # Create a DataFrame
    df = pd.DataFrame({
        'time_delta': time_deltas,
        'cumulative_not_running': cumulative_not_running
    })

    # Define the models
    models = [
        LinearRegression(),
        SVR(),
        DecisionTreeRegressor()
    ]

    future_time_delta = time_deltas[-1] + 7200
    predictions3 = []

    # Train each model and make predictions
    for model in models:
        model.fit(df[['time_delta']], df['cumulative_not_running'])
        predicted_cumulative_downtime = model.predict([[future_time_delta]])[0]
        current_cumulative_downtime = df['cumulative_not_running'].iloc[-1]
        predicted_downtime_next_2_hours = predicted_cumulative_downtime - current_cumulative_downtime
        predictions3.append(predicted_downtime_next_2_hours)

    # Calculate the average prediction
    final_prediction2 = sum(predictions3) / len(predictions3)

    # Query the event data for the current user
    events = Event.query.filter_by(user_id=current_user.id).order_by(Event.timestamp).all()

    if not events:
        flash('No events data available for prediction', category='error')
        return render_template('prediction.html', graph_json=None)

    # Prepare data for the classification model
    timestamps = [event.timestamp for event in events]
    reasons = [event.reason for event in events]

    # Convert timestamps to numeric values (e.g., seconds since the first timestamp)
    time_deltas = [(timestamp - timestamps[0]).total_seconds() for timestamp in timestamps]

    # Create a DataFrame
    df = pd.DataFrame({
        'time_delta': time_deltas,
        'reason': reasons
    })

    # Encode the reasons
    reason_map = {reason: idx for idx, reason in enumerate(df['reason'].unique())}
    df['reason'].replace(reason_map, inplace=True)

    # Prepare sequences for the LSTM
    def create_sequences(data, time_step):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step)])
            y.append(data[i + time_step])
        return np.array(X), np.array(y)

    time_step = 10  # Number of past time steps to use for prediction
    dataset = df[['reason']].values

    if len(dataset) <= time_step:
        flash('Not enough data to create sequences', category='error')
        return render_template('prediction.html', graph_json=None)

    X, y = create_sequences(dataset, time_step)

    if X.size == 0 or y.size == 0:
        flash('Failed to create valid sequences for prediction', category='error')
        return render_template('prediction.html', graph_json=None)

    # Reshape data to fit LSTM input shape
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # One-hot encode the target variable
    y = tf.keras.utils.to_categorical(y, num_classes=len(reason_map))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(len(reason_map), activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=1, verbose=2)

    # Predict the reason distribution after 2 hours
    future_sequence = np.tile(dataset[-time_step:], (1, 1)).reshape((1, time_step, 1))
    future_reason_distribution = model.predict(future_sequence)[0]

    # Decode the predicted reason distribution
    decoded_distribution = {reason: future_reason_distribution[idx] for reason, idx in reason_map.items()}

    # Prepare data for the doughnut chart
    labels = list(decoded_distribution.keys())
    values = list(decoded_distribution.values())
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_layout(
        title='Distribution After 2 Hours',
        showlegend=True
    )

    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Query events for the current user
    events = Event.query.filter_by(user_id=current_user.id).all()

    if not events:
        return render_template('capacity_utilization.html', graph_json=None)

    # Calculate capacity utilization for each event type
    event_types = set(event.event_type for event in events)
    utilization_data = {}
    for event_type in event_types:
        event_type_events = [event for event in events if event.event_type == event_type]
        total_capacity = sum(event.capacity for event in event_type_events)
        utilized_capacity = sum(event.capacity for event in event_type_events if event.reason != 'Not Running')
        utilization_percentage = (utilized_capacity / total_capacity) * 100 if total_capacity > 0 else 0
        utilization_data[event_type] = utilization_percentage

    # Define a color palette for the bars
    colors = px.colors.qualitative.Plotly

    # Create a bar chart with different colors for each bar
    data = []
    for i, (event_type, utilization_percentage) in enumerate(utilization_data.items()):
        data.append(go.Bar(
            x=[event_type],
            y=[utilization_percentage],
            name=event_type,
            marker=dict(color=colors[i])
        ))

    layout = go.Layout(
        title='Capacity Utilization by Event Type',
        xaxis=dict(title='Event Type'),
        yaxis=dict(title='Capacity Utilization (%)'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        barmode='group'
    )

    fig = go.Figure(data=data, layout=layout)
    graph_json2 = fig.to_json()

    # Query events for the current user
    events = Event.query.filter_by(user_id=current_user.id).all()

    # Prepare the data
    df = pd.DataFrame([{
        'event_type': event.event_type,
        'operator': event.operator
    } for event in events])

    if df.empty:
        return render_template('predict_event_type.html', predictions=None, report=None)

    # Encode the categorical variables
    label_encoder_event = LabelEncoder()
    label_encoder_operator = LabelEncoder()

    df['event_type'] = label_encoder_event.fit_transform(df['event_type'])
    df['operator'] = label_encoder_operator.fit_transform(df['operator'])

    X = df[['operator']].values
    y = df['event_type'].values

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2)

    # Make predictions
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Classification report
    target_names = label_encoder_event.inverse_transform([0, 1])
    report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)

    prediction = pd.DataFrame({
        'Actual': label_encoder_event.inverse_transform(y_test),
        'Predicted': label_encoder_event.inverse_transform(y_pred)
    })

    return render_template('prediction.html', 
                           predictions=predictions, 
                           predictions2={'Final Prediction': final_prediction}, 
                           predictions3={'Final Prediction': final_prediction2},
                           graph_json=graph_json,
                           graph_json2=graph_json2)


@views.route('/statistics')
@login_required
def statistics():
    # Query event types and their frequencies for the current user
    event_types = db.session.query(Event.event_type, func.count(Event.event_type)).filter_by(user_id=current_user.id).group_by(Event.event_type).all()

    # Convert query result to dictionary for easy access in the template
    event_type_counts = {event_type: count for event_type, count in event_types}

    # Query all events for the current user
    events = Event.query.filter_by(user_id=current_user.id).order_by(Event.timestamp).all()

    running_duration = timedelta(seconds=0)
    not_running_duration = timedelta(seconds=0)

    # Iterate through events to calculate durations
    for i in range(len(events)):
        if i < len(events) - 1:
            event = events[i]
            next_event = events[i + 1]

            # Calculate duration between current event and next event
            duration = next_event.timestamp - event.timestamp

            if event.event_type == '1':  # '1' represents 'running'
                running_duration += duration
            elif event.event_type == '0':  # '0' represents 'not running'
                not_running_duration += duration

    # Convert running and not running durations to total seconds
    running_seconds = int(running_duration.total_seconds())
    not_running_seconds = int(not_running_duration.total_seconds())

    # Convert total seconds to formatted time strings
    running_time_str = str(timedelta(seconds=running_seconds))
    not_running_time_str = str(timedelta(seconds=not_running_seconds))

    # Query all events for the current user
    events = Event.query.filter_by(user_id=current_user.id).order_by(Event.timestamp).all()

    breakdown_duration = timedelta(seconds=0)
    lack_of_materials_duration = timedelta(seconds=0)
    lack_of_operators_duration = timedelta(seconds=0)

    # Iterate through events to calculate durations
    for i in range(len(events)):
        if i < len(events) - 1:
            event = events[i]
            next_event = events[i + 1]

            # Calculate duration between current event and next event
            duration = next_event.timestamp - event.timestamp

            if event.reason == 'Breakdown':  # '1' represents 'running'
                breakdown_duration += duration
            elif event.reason == 'Lack of materials':  # '0' represents 'not running'
                lack_of_materials_duration += duration
            elif event.reason == 'Lack of operators':  # '0' represents 'not running'
                lack_of_operators_duration += duration

    # Convert running and not running durations to total seconds
    breakdown_seconds = int(breakdown_duration.total_seconds())
    lack_of_materials_seconds = int(lack_of_materials_duration.total_seconds())
    lack_of_operators_seconds = int(lack_of_operators_duration.total_seconds())

    # Convert total seconds to formatted time strings
    breakdown_time_str = str(timedelta(seconds=breakdown_seconds))
    lack_of_materials_time_str = str(timedelta(seconds=lack_of_materials_seconds))
    lack_of_operators_time_str = str(timedelta(seconds=lack_of_operators_seconds))


    # Query event types and their frequencies for the current user
    event_types = db.session.query(Event.event_type, func.count(Event.event_type)).filter_by(user_id=current_user.id).group_by(Event.event_type).all()

    # Convert query result to dictionary for easy access in the template
    event_type_counts = {event_type: count for event_type, count in event_types}

    # Query all events for the current user
    events = Event.query.filter_by(user_id=current_user.id).order_by(Event.timestamp).all()

    # Initialize a dictionary to store running durations per operator
    running_durations_per_operator = {'1': timedelta(seconds=0), '2': timedelta(seconds=0), '3': timedelta(seconds=0)}

    # Iterate through events to calculate durations
    for i in range(len(events)):
        if i < len(events) - 1:
            event = events[i]
            next_event = events[i + 1]

            # Calculate duration between current event and next event
            duration = next_event.timestamp - event.timestamp

            # Check if the event type is 'running'
            if event.event_type == '1':  # '1' represents 'running'
                operator = event.operator
                if operator in running_durations_per_operator:
                    running_durations_per_operator[operator] += duration

    # Convert running durations per operator to total seconds and formatted time strings
    running_times_str_per_operator = {operator: str(timedelta(seconds=int(duration.total_seconds()))) for operator, duration in running_durations_per_operator.items()}


    # Query event types and their frequencies for the current user
    event_types = db.session.query(Event.event_type, func.count(Event.event_type)).filter_by(user_id=current_user.id).group_by(Event.event_type).all()

    # Convert query result to dictionary for easy access in the template
    event_type_counts = {event_type: count for event_type, count in event_types}

    # Query all events for the current user
    events = Event.query.filter_by(user_id=current_user.id).order_by(Event.timestamp).all()

    # Initialize dictionaries to store running and not_running durations per operator
    running_durations_per_operator = {'1': timedelta(seconds=0), '2': timedelta(seconds=0), '3': timedelta(seconds=0)}
    not_running_durations_per_operator = {'1': timedelta(seconds=0), '2': timedelta(seconds=0), '3': timedelta(seconds=0)}

    # Iterate through events to calculate durations
    for i in range(len(events)):
        if i < len(events) - 1:
            event = events[i]
            next_event = events[i + 1]

            # Calculate duration between current event and next event
            duration = next_event.timestamp - event.timestamp

            # Check if the event type is 'running' or 'not_running'
            if event.event_type == '1':  # '1' represents 'running'
                operator = event.operator
                if operator in running_durations_per_operator:
                    running_durations_per_operator[operator] += duration
            elif event.event_type == '0':  # '0' represents 'not_running'
                operator = event.operator
                if operator in not_running_durations_per_operator:
                    not_running_durations_per_operator[operator] += duration

    # Convert running and not_running durations per operator to total seconds and formatted time strings
    running_times_str_per_operator = {operator: str(timedelta(seconds=int(duration.total_seconds()))) for operator, duration in running_durations_per_operator.items()}
    not_running_times_str_per_operator = {operator: str(timedelta(seconds=int(duration.total_seconds()))) for operator, duration in not_running_durations_per_operator.items()}

    # Prepare data for bar graphs
    operators = list(running_durations_per_operator.keys())
    running_seconds = [int(duration.total_seconds()) for duration in running_durations_per_operator.values()]
    not_running_seconds = [int(duration.total_seconds()) for duration in not_running_durations_per_operator.values()]

    # Define colors for the bars
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Create bar graphs using Plotly
    not_running_bar_data = [
        go.Bar(
            x=operators,
            y=not_running_seconds,
            name='Not Running Duration',
            marker=dict(color=colors)  # Set colors for each bar
        )
    ]

    running_bar_data = [
        go.Bar(
            x=operators,
            y=running_seconds,
            name='Running Duration',
            marker=dict(color=colors)  # Set colors for each bar
        )
    ]

    # Layout for not running bar graph
    not_running_bar_layout = go.Layout(
        title='Not Running Duration per Operator',
        xaxis=dict(title='Operator'),
        yaxis=dict(title='Duration (seconds)'),
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Layout for running bar graph
    running_bar_layout = go.Layout(
        title='Running Duration per Operator',
        xaxis=dict(title='Operator'),
        yaxis=dict(title='Duration (seconds)'),
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Serialize the chart data to JSON
    not_running_bar_chart_data = {
        'data': json.dumps(not_running_bar_data, cls=plotly.utils.PlotlyJSONEncoder),
        'layout': json.dumps(not_running_bar_layout, cls=plotly.utils.PlotlyJSONEncoder)
    }

    running_bar_chart_data = {
        'data': json.dumps(running_bar_data, cls=plotly.utils.PlotlyJSONEncoder),
        'layout': json.dumps(running_bar_layout, cls=plotly.utils.PlotlyJSONEncoder)
    }


    return render_template('statistics.html', 
                           event_type_counts=event_type_counts, 
                           running_time=running_time_str, 
                           not_running_time=not_running_time_str, 
                           lack_time=lack_of_materials_time_str, 
                           breakdown_time=breakdown_time_str, 
                           lack_of_operators_time=lack_of_operators_time_str,
                           running_times_str_per_operator=running_times_str_per_operator,
                           not_running_bar_chart_data=not_running_bar_chart_data,
                           running_bar_chart_data=running_bar_chart_data)


@views.route('/capacity')
@login_required
def capacity():
    # Query capacity and timestamps from events for the current user
    events = Event.query.filter_by(user_id=current_user.id).order_by(Event.timestamp).all()

    timestamps = [event.timestamp.strftime('%Y-%m-%d %H:%M:%S') for event in events]  # Convert datetime to string
    capacities = [event.capacity for event in events]

    # Calculate statistical measures
    mean_capacity = np.mean(capacities)
    median_capacity = np.median(capacities)
    max_capacity = np.max(capacities)
    min_capacity = np.min(capacities)

    # Create a bar chart using Plotly
    data = [
        go.Bar(
            x=timestamps,
            y=capacities,
            name='Capacity'
        )
    ]

    layout = go.Layout(
        title='Capacity Over Time',
        xaxis=dict(title='Timestamp'),
        yaxis=dict(title='Capacity'),
        barmode='group',  # Set bar mode to group for clustered bars
        plot_bgcolor='white',  # Set the background color of the plot area
        paper_bgcolor='white'
    )

    # Serialize the chart data to JSON
    chart_data = {
        'data': json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder),  # Serialize data using PlotlyJSONEncoder
        'layout': json.dumps(layout, cls=plotly.utils.PlotlyJSONEncoder),  # Serialize layout using PlotlyJSONEncoder
        'mean_capacity': mean_capacity,
        'median_capacity': median_capacity,
        'max_capacity': max_capacity,
        'min_capacity': min_capacity
    }

    return render_template("capacity.html", chart_data=chart_data)


@views.route('/', methods=['GET', 'POST'])
@login_required
def home():

    # Query all reasons and their counts from the Event model
    reasons_count = db.session.query(Event.reason, func.count(Event.reason)).group_by(Event.reason).all()
    
    # Extract reasons and counts into separate lists for plotting
    reasons = [reason for reason, _ in reasons_count]
    counts = [count for _, count in reasons_count]

    # Prepare data as JSON to pass to the template
    chart_data = json.dumps({'reasons': reasons, 'counts': counts})


    # Query event types and their counts from the Event model
    event_types_count = db.session.query(Event.event_type, func.count(Event.event_type)).group_by(Event.event_type).all()

    # Map event types to meaningful labels for the pie chart
    labels = ['Not Running', 'Running']
    counts = [0, 0]  # Initialize counts for 'not running' and 'running'

    # Process event type counts
    for event_type, count in event_types_count:
        if event_type == '1':  # '1' represents 'running'
            counts[1] = count
        elif event_type == '0':  # '0' represents 'not running'
            counts[0] = count

    # Prepare data as JSON to pass to the template
    event_type = json.dumps({'labels': labels, 'counts': counts})

    
    # Query all events for the current user
    events = Event.query.filter_by(user_id=current_user.id).order_by(Event.timestamp).all()

    running_duration = timedelta(seconds=0)
    not_running_duration = timedelta(seconds=0)

    # Iterate through events to calculate durations
    for i in range(len(events)):
        if i < len(events) - 1:
            event = events[i]
            next_event = events[i + 1]

            # Calculate duration between current event and next event
            duration = next_event.timestamp - event.timestamp

            if event.event_type == '1':  # '1' represents 'running'
                running_duration += duration
            elif event.event_type == '0':  # '0' represents 'not running'
                not_running_duration += duration

    # Convert running and not running durations to total seconds
    running_seconds = int(running_duration.total_seconds())
    not_running_seconds = int(not_running_duration.total_seconds())

    # Convert total seconds to formatted time strings
    running_time_str = str(timedelta(seconds=running_seconds))
    not_running_time_str = str(timedelta(seconds=not_running_seconds))


    return render_template("home.html", user=current_user, chart_data=chart_data, event_type=event_type, running_time=running_time_str, not_running_time=not_running_time_str)

    
    """if request.method == 'POST':
        
        capacity = request.form.get('capacity')
        event_type = request.form.get('event_type')
        operator = request.form.get('operator')
        reason = request.form.get('reason')
        now = datetime.utcnow()
        if len(capacity) < 1:
            flash('Capacity is too small!', category='error')
        else:
            new_note = Event(capacity=capacity, timestamp=now, event_type=event_type, operator=operator, reason=reason, user_id=current_user.id)
            db.session.add(new_note)
            db.session.commit()
            flash('Note added.', category='success')
        flash('Event added with timestamp.', category='success')


    return render_template("home.html", user=current_user)"""


"""@views.route('/update/<int:id>', methods=['GET', 'POST'])
def update(id):
    note_to_update = Note.query.get_or_404(id)
    if request.method == "POST":
        note_to_update.data = request.form['data']
        try:
            db.session.commit()
            return redirect('/')
        except:
            return "There was an error"
    return render_template("update.html", note_to_update=note_to_update, user=current_user)

"""


"""@views.route('/delete-note', methods=['POST'])
def delete_note():
    note = json.loads(request.data)
    noteId = note['noteId']
    note = Note.query.get(noteId)
    if note:
        if note.user_id == current_user.id:
            db.session.delete(note)
            db.session.commit()
            
    return jsonify({})"""
