from flask import Blueprint, render_template, request, flash, jsonify, redirect
from flask_login import login_required, current_user
from .models import Note
from . import db
import json
import pandas as pd
import plotly
import plotly.express as px

views = Blueprint('views', __name__)

@views.route('/update/<int:id>', methods=['GET', 'POST'])
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


@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    """if request.method == 'POST':
        note = request.form.get('note')

        if len(note) < 1:
            flash('Note is too short!', category='error')
        else:
            new_note = Note(data=note, user_id=current_user.id)
            db.session.add(new_note)
            db.session.commit()
            flash('Note added.', category='success')"""

    df = pd.DataFrame([
        dict(Task="Job A", Start='2009-01-01', Finish='2009-02-28', Resource="Alex"),
        dict(Task="Job B", Start='2009-03-05', Finish='2009-04-15', Resource="Alex"),
        dict(Task="Job C", Start='2009-02-20', Finish='2009-05-30', Resource="Max")
    ])

    fig1 = px.timeline(df, x_start="Start", x_end="Finish", y="Resource", color="Resource")

    graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template("home.html", user=current_user, graph1JSON = graph1JSON)

@views.route('/delete-note', methods=['POST'])
def delete_note():
    note = json.loads(request.data)
    noteId = note['noteId']
    note = Note.query.get(noteId)
    if note:
        if note.user_id == current_user.id:
            db.session.delete(note)
            db.session.commit()
            
    return jsonify({})