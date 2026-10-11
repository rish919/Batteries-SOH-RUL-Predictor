"""Plotly chart constructors shared by dashboard sections."""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

COLORS=['#1d4ed8','#0891b2','#059669','#d97706','#7c3aed','#dc2626']

def line_chart(frame,x,y,title,y_title):
    fig=px.line(frame,x=x,y=y,color='battery_id' if 'battery_id' in frame else None,
                color_discrete_sequence=COLORS,markers=False,labels={x:'Discharge cycle',y:y_title},title=title)
    fig.update_layout(height=400,legend_title='Battery',margin=dict(l=12,r=12,t=55,b=12),hovermode='x unified')
    fig.update_xaxes(rangemode='tozero',gridcolor='#e5e7eb');fig.update_yaxes(gridcolor='#e5e7eb')
    return fig

def scatter_actual(predictions,title,units):
    fig=px.scatter(predictions,x='actual',y='predicted',color='model',symbol='battery_id',opacity=.68,
                   labels={'actual':f'Actual ({units})','predicted':f'OOF prediction ({units})'},title=title)
    lo=min(predictions.actual.min(),predictions.predicted.min());hi=max(predictions.actual.max(),predictions.predicted.max())
    fig.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode='lines',name='Ideal',line=dict(color='#111827',dash='dash')))
    fig.update_layout(height=480,margin=dict(l=12,r=12,t=55,b=12));return fig

def residual_chart(frame,units):
    p=frame.copy();p['residual']=p.predicted-p.actual
    return px.scatter(p,x='actual',y='residual',color='model',opacity=.6,
        labels={'actual':f'Actual ({units})','residual':f'Prediction error ({units})'},title='Held-out residuals')
