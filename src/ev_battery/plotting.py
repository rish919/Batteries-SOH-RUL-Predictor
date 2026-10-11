"""Static exploratory plots from observed measurements only."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def exploratory_figures(df: pd.DataFrame, directory: Path) -> None:
    directory.mkdir(parents=True,exist_ok=True)
    for column,label,name in [('capacity_ah','Measured discharge capacity (Ah)','capacity_degradation'),('soh_pct','SOH (%)','soh_trajectory')]:
        fig,ax=plt.subplots(figsize=(11,6))
        for bid,g in df.groupby('battery_id'): ax.plot(g.cycle_index,g[column],label=bid,linewidth=1)
        ax.set(xlabel='Discharge cycle index',ylabel=label,title=label+' across recorded batteries')
        ax.legend(ncol=4,fontsize=7,bbox_to_anchor=(1.01,1),loc='upper left'); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(directory/f'{name}.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,col,label in zip(axes,['discharge_voltage_mean','discharge_current_mean','discharge_temperature_mean'],['Mean voltage (V)','Mean current (A)','Mean temperature (°C)']):
        p=ax.scatter(df[col],df.capacity_ah,c=df.cycle_index,s=8,alpha=.5,cmap='viridis')
        ax.set(xlabel=label,ylabel='Capacity (Ah)'); ax.grid(alpha=.2)
    fig.colorbar(p,ax=axes.tolist(),label='Discharge cycle index',fraction=.02)
    fig.suptitle('Observed feature relationships (descriptive, not model validation)')
    fig.savefig(directory/'feature_relationships.png',dpi=160,bbox_inches='tight'); plt.close(fig)
