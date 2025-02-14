from app import db
from datetime import datetime

class standardLog(db.Model):
    __tablename__ = 'standardlog'  
    __table_args__ = {'extend_existing': True} 
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class breakLog(db.Model):
    __tablename__ = 'breaklog'
    __table_args__ = {'extend_existing': True} 
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class NormalLog(db.Model):
    __tablename__ = 'normallog'
    __table_args__ = {'extend_existing': True} 
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class DailyCount(db.Model):
    __tablename__ = 'dailycount'
    __table_args__ = {'extend_existing': True} 
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    date = db.Column(db.Date, unique=True, nullable=False, default=datetime.utcnow().date)
    final_standard_count = db.Column(db.Integer, nullable=False, default=0)
    final_break_count = db.Column(db.Integer, nullable=False, default=0)
    final_normal_count = db.Column(db.Integer, nullable=False, default=0)
    
class paper_size(db.Model):
    __tablename__ = 'paper_size'
    __table_args__ = {'extend_existing': True} 
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    width_height = db.Column(db.String(10), nullable=False)
    tolerance_cm = db.Column(db.Integer, nullable=False, default=0)
    standard_paper_size_cm = db.Column(db.Integer, nullable=False, default=0)
    pixel_to_cm = db.Column(db.Float, nullable=False, default=0)
