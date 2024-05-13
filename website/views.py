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


views = Blueprint('views', __name__)

from datetime import datetime, timedelta

@views.route('/capacity')
@login_required
def capacity():
    # Query capacity and timestamps from events for the current user
    events = Event.query.filter_by(user_id=current_user.id).order_by(Event.timestamp).all()

    timestamps = [event.timestamp.strftime('%Y-%m-%d %H:%M:%S') for event in events]  # Convert datetime to string
    capacities = [event.capacity for event in events]

    # Create a line chart using Plotly
    data = [
        go.Scatter(
            x=timestamps,
            y=capacities,
            mode='lines',
            name='Capacity'
        )
    ]

    layout = go.Layout(
        title='Capacity Over Time',
        xaxis=dict(title='Timestamp'),
        yaxis=dict(title='Capacity'),
        plot_bgcolor='black',  # Set the background color of the plot area
        paper_bgcolor='black'
    )

    # Serialize the chart data to JSON
    chart_data = {
        'data': json.dumps([trace.to_plotly_json() for trace in data]),  # Convert each trace to Plotly JSON format
        'layout': json.dumps(layout, cls=plotly.utils.PlotlyJSONEncoder)  # Serialize layout using PlotlyJSONEncoder
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
