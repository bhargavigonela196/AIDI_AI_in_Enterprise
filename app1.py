from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
import os

# Initialize Flask App
app = Flask(__name__)

# Database Configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.path.join(BASE_DIR, 'database.db')}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize Database and Marshmallow
db = SQLAlchemy(app)
ma = Marshmallow(app)

# Define Student Model
class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    dob = db.Column(db.String(10), nullable=False)  # Stored as 'YYYY-MM-DD'
    amount_due = db.Column(db.Float, nullable=False)

    def __repr__(self):
        return f"<Student {self.first_name} {self.last_name}>"

# Define Student Schema for Serialization
class StudentSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Student
        load_instance = True

student_schema = StudentSchema()
students_schema = StudentSchema(many=True)

# Routes
@app.route('/students', methods=['POST'])
def add_student():
    data = request.json
    new_student = Student(
        first_name=data['first_name'],
        last_name=data['last_name'],
        dob=data['dob'],
        amount_due=data['amount_due']
    )
    db.session.add(new_student)
    db.session.commit()
    results = {"students": student_schema.dump(new_student), "message": "Student created successfully"}
    return jsonify(results), 201

@app.route('/students', methods=['GET'])
def get_students():
    students = Student.query.all()
    students = [student_schema.dump(student) for student in students]
    resutls = {"students": students, "message": "Student details fetched successfully"}
    return jsonify(resutls), 200

@app.route('/students/<int:id>', methods=['GET'])
def get_student(id):
    student = Student.query.get_or_404(id)
    resutls = {"students": student_schema.dump(student), "message": "Student details fetched successfully"}
    return jsonify(resutls), 200

@app.route('/students/<int:id>', methods=['PUT'])
def update_student(id):
    student = Student.query.get_or_404(id)
    data = request.json
    student.first_name = data.get('first_name', student.first_name)
    student.last_name = data.get('last_name', student.last_name)
    student.dob = data.get('dob', student.dob)
    student.amount_due = data.get('amount_due', student.amount_due)

    db.session.commit()
    resutls = {"students": student_schema.dump(student), "message": "Student details updated successfully"}
    return jsonify(resutls), 200

@app.route('/students/<int:id>', methods=['DELETE'])
def delete_student(id):
    student = Student.query.get_or_404(id)
    db.session.delete(student)
    db.session.commit()
    return jsonify({"id": id, "message": "Student deleted successfully"}), 200

# Run Flask App
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
