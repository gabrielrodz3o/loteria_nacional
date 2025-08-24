"""SQLAlchemy database models for lottery prediction system."""

from sqlalchemy import Column, Integer, String, Float, Boolean, Date, Time, Text, ForeignKey, DateTime, CheckConstraint, UniqueConstraint, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from config.database import Base
from datetime import datetime


class TipoLoteria(Base):
    """Lottery types model."""
    __tablename__ = "tipos_loteria"
    
    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String(100), unique=True, nullable=False)
    descripcion = Column(Text)
    hora_sorteo = Column(Time, nullable=False)
    activo = Column(Boolean, default=True)
    creado_en = Column(DateTime, default=func.now())
    
    # Relationships
    sorteos = relationship("Sorteo", back_populates="tipo_loteria")
    predicciones_quiniela = relationship("PrediccionQuiniela", back_populates="tipo_loteria")
    predicciones_pale = relationship("PrediccionPale", back_populates="tipo_loteria")
    predicciones_tripleta = relationship("PrediccionTripleta", back_populates="tipo_loteria")


class TipoJuego(Base):
    """Game types model."""
    __tablename__ = "tipos_juego"
    
    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String(50), unique=True, nullable=False)
    descripcion = Column(Text)
    formato_numeros = Column(String(100), nullable=False)
    activo = Column(Boolean, default=True)
    creado_en = Column(DateTime, default=func.now())
    
    # Relationships
    resultados_predicciones = relationship("ResultadoPrediccion", back_populates="tipo_juego")


class Sorteo(Base):
    """Historical lottery draws model."""
    __tablename__ = "sorteos"
    
    id = Column(Integer, primary_key=True, index=True)
    fecha = Column(Date, nullable=False)
    tipo_loteria_id = Column(Integer, ForeignKey("tipos_loteria.id"), nullable=False)
    primer_lugar = Column(Integer, nullable=False)
    segundo_lugar = Column(Integer, nullable=False)
    tercer_lugar = Column(Integer, nullable=False)
    fuente_scraping = Column(String(255))
    creado_en = Column(DateTime, default=func.now())
    
    # Constraints
    __table_args__ = (
        CheckConstraint('primer_lugar >= 0 AND primer_lugar <= 99', name='check_primer_lugar_range'),
        CheckConstraint('segundo_lugar >= 0 AND segundo_lugar <= 99', name='check_segundo_lugar_range'),
        CheckConstraint('tercer_lugar >= 0 AND tercer_lugar <= 99', name='check_tercer_lugar_range'),
        UniqueConstraint('fecha', 'tipo_loteria_id', name='unique_sorteo_fecha_tipo'),
        Index('idx_sorteos_fecha', 'fecha'),
        Index('idx_sorteos_tipo_loteria', 'tipo_loteria_id'),
        Index('idx_sorteos_fecha_tipo', 'fecha', 'tipo_loteria_id'),
        Index('idx_sorteos_combinacion_tipo', 'tipo_loteria_id', 'primer_lugar', 'segundo_lugar', 'tercer_lugar'),
    )
    
    # Relationships
    tipo_loteria = relationship("TipoLoteria", back_populates="sorteos")
    vectores = relationship("Vector", back_populates="sorteo", cascade="all, delete-orphan")


class PrediccionQuiniela(Base):
    """Daily quiniela predictions model."""
    __tablename__ = "predicciones_quiniela"
    
    id = Column(Integer, primary_key=True, index=True)
    fecha_prediccion = Column(Date, nullable=False)
    tipo_loteria_id = Column(Integer, ForeignKey("tipos_loteria.id"), nullable=False)
    posicion = Column(Integer, nullable=False)
    numero_predicho = Column(Integer, nullable=False)
    probabilidad = Column(Float)
    metodo_generacion = Column(String(100), nullable=False)
    score_confianza = Column(Float)
    creado_en = Column(DateTime, default=func.now())
    
    # Constraints
    __table_args__ = (
        CheckConstraint('posicion IN (1, 2, 3)', name='check_quiniela_posicion'),
        CheckConstraint('numero_predicho >= 0 AND numero_predicho <= 99', name='check_quiniela_numero_range'),
        CheckConstraint('probabilidad >= 0 AND probabilidad <= 1', name='check_quiniela_probabilidad'),
        CheckConstraint('score_confianza >= 0 AND score_confianza <= 1', name='check_quiniela_confianza'),
        UniqueConstraint('fecha_prediccion', 'tipo_loteria_id', 'posicion', name='unique_quiniela_fecha_tipo_posicion'),
        Index('idx_quiniela_fecha', 'fecha_prediccion'),
        Index('idx_quiniela_tipo_loteria', 'tipo_loteria_id'),
        Index('idx_quiniela_posicion', 'posicion'),
        Index('idx_quiniela_numero', 'numero_predicho'),
        Index('idx_quiniela_metodo', 'metodo_generacion'),
    )
    
    # Relationships
    tipo_loteria = relationship("TipoLoteria", back_populates="predicciones_quiniela")


class PrediccionPale(Base):
    """Daily pale predictions model."""
    __tablename__ = "predicciones_pale"
    
    id = Column(Integer, primary_key=True, index=True)
    fecha_prediccion = Column(Date, nullable=False)
    tipo_loteria_id = Column(Integer, ForeignKey("tipos_loteria.id"), nullable=False)
    posicion = Column(Integer, nullable=False)
    numero_1 = Column(Integer, nullable=False)
    numero_2 = Column(Integer, nullable=False)
    probabilidad = Column(Float)
    metodo_generacion = Column(String(100), nullable=False)
    score_confianza = Column(Float)
    creado_en = Column(DateTime, default=func.now())
    
    # Constraints
    __table_args__ = (
        CheckConstraint('posicion IN (1, 2, 3)', name='check_pale_posicion'),
        CheckConstraint('numero_1 >= 0 AND numero_1 <= 99', name='check_pale_numero1_range'),
        CheckConstraint('numero_2 >= 0 AND numero_2 <= 99', name='check_pale_numero2_range'),
        CheckConstraint('numero_1 != numero_2', name='pale_numeros_diferentes'),
        CheckConstraint('probabilidad >= 0 AND probabilidad <= 1', name='check_pale_probabilidad'),
        CheckConstraint('score_confianza >= 0 AND score_confianza <= 1', name='check_pale_confianza'),
        UniqueConstraint('fecha_prediccion', 'tipo_loteria_id', 'posicion', name='unique_pale_fecha_tipo_posicion'),
        Index('idx_pale_fecha', 'fecha_prediccion'),
        Index('idx_pale_tipo_loteria', 'tipo_loteria_id'),
        Index('idx_pale_posicion', 'posicion'),
        Index('idx_pale_numeros', 'numero_1', 'numero_2'),
        Index('idx_pale_metodo', 'metodo_generacion'),
    )
    
    # Relationships
    tipo_loteria = relationship("TipoLoteria", back_populates="predicciones_pale")


class PrediccionTripleta(Base):
    """Daily tripleta predictions model."""
    __tablename__ = "predicciones_tripleta"
    
    id = Column(Integer, primary_key=True, index=True)
    fecha_prediccion = Column(Date, nullable=False)
    tipo_loteria_id = Column(Integer, ForeignKey("tipos_loteria.id"), nullable=False)
    posicion = Column(Integer, nullable=False)
    numero_1 = Column(Integer, nullable=False)
    numero_2 = Column(Integer, nullable=False)
    numero_3 = Column(Integer, nullable=False)
    probabilidad = Column(Float)
    metodo_generacion = Column(String(100), nullable=False)
    score_confianza = Column(Float)
    creado_en = Column(DateTime, default=func.now())
    
    # Constraints
    __table_args__ = (
        CheckConstraint('posicion IN (1, 2, 3)', name='check_tripleta_posicion'),
        CheckConstraint('numero_1 >= 0 AND numero_1 <= 99', name='check_tripleta_numero1_range'),
        CheckConstraint('numero_2 >= 0 AND numero_2 <= 99', name='check_tripleta_numero2_range'),
        CheckConstraint('numero_3 >= 0 AND numero_3 <= 99', name='check_tripleta_numero3_range'),
        CheckConstraint('numero_1 != numero_2 AND numero_1 != numero_3 AND numero_2 != numero_3', name='tripleta_numeros_diferentes'),
        CheckConstraint('probabilidad >= 0 AND probabilidad <= 1', name='check_tripleta_probabilidad'),
        CheckConstraint('score_confianza >= 0 AND score_confianza <= 1', name='check_tripleta_confianza'),
        UniqueConstraint('fecha_prediccion', 'tipo_loteria_id', 'posicion', name='unique_tripleta_fecha_tipo_posicion'),
        Index('idx_tripleta_fecha', 'fecha_prediccion'),
        Index('idx_tripleta_tipo_loteria', 'tipo_loteria_id'),
        Index('idx_tripleta_posicion', 'posicion'),
        Index('idx_tripleta_numeros', 'numero_1', 'numero_2', 'numero_3'),
        Index('idx_tripleta_metodo', 'metodo_generacion'),
    )
    
    # Relationships
    tipo_loteria = relationship("TipoLoteria", back_populates="predicciones_tripleta")


class MetodoPrediccion(Base):
    """Prediction methods model."""
    __tablename__ = "metodos_prediccion"
    
    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String(100), unique=True, nullable=False)
    descripcion = Column(Text)
    version = Column(String(50))
    parametros = Column(JSONB)
    activo = Column(Boolean, default=True)
    creado_en = Column(DateTime, default=func.now())


class Vector(Base):
    """Vector embeddings model using pgvector."""
    __tablename__ = "vectores"
    
    id = Column(Integer, primary_key=True, index=True)
    sorteo_id = Column(Integer, ForeignKey("sorteos.id", ondelete="CASCADE"), nullable=False)
    embedding = Column(Vector(128), nullable=False)
    creado_en = Column(DateTime, default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_vectores_sorteo_id', 'sorteo_id'),
        # HNSW index for vector similarity search
        Index('idx_vectores_embedding_hnsw', 'embedding', postgresql_using='hnsw', postgresql_with={'m': 16, 'ef_construction': 64}),
    )
    
    # Relationships
    sorteo = relationship("Sorteo", back_populates="vectores")


class ResultadoPrediccion(Base):
    """Prediction results evaluation model."""
    __tablename__ = "resultados_predicciones"
    
    id = Column(Integer, primary_key=True, index=True)
    fecha_sorteo = Column(Date, nullable=False)
    tipo_loteria_id = Column(Integer, ForeignKey("tipos_loteria.id"), nullable=False)
    tipo_juego_id = Column(Integer, ForeignKey("tipos_juego.id"), nullable=False)
    prediccion_id = Column(Integer)  # Generic reference to prediction ID
    acierto = Column(Boolean, nullable=False)
    tipo_acierto = Column(String(50))  # 'exacto', 'parcial', 'ninguno'
    puntos_obtenidos = Column(Integer, default=0)
    creado_en = Column(DateTime, default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_resultados_fecha', 'fecha_sorteo'),
        Index('idx_resultados_tipo_loteria', 'tipo_loteria_id'),
        Index('idx_resultados_tipo_juego', 'tipo_juego_id'),
        Index('idx_resultados_acierto', 'acierto'),
    )
    
    # Relationships
    tipo_juego = relationship("TipoJuego", back_populates="resultados_predicciones")