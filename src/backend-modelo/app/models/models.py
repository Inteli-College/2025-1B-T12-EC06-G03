from datetime import datetime
from sqlalchemy.dialects.postgresql import JSON, DOUBLE_PRECISION
from app import db

class Usuario(db.Model):
    __tablename__ = 'usuarios'
    id             = db.Column(db.BigInteger, primary_key=True)
    nome           = db.Column(db.Text)
    email          = db.Column(db.Text, unique=True)
    senha          = db.Column(db.Text)
    cargo          = db.Column(db.Text)
    data_criacao   = db.Column(db.DateTime, default=datetime.utcnow)
    ultimo_acesso  = db.Column(db.DateTime)

    relatorios          = db.relationship('Relatorio',       back_populates='usuario', cascade='all, delete-orphan')
    logs_alteracoes     = db.relationship('LogAlteracao',    back_populates='usuario', cascade='all, delete-orphan')
    responsaveis        = db.relationship('ResponsavelProjeto', back_populates='usuario', cascade='all, delete-orphan')

class Empresa(db.Model):
    __tablename__ = 'empresas'
    id         = db.Column(db.BigInteger, primary_key=True)
    nome       = db.Column(db.Text)
    cnpj       = db.Column(db.Text, unique=True)
    endereco   = db.Column(db.Text)
    telefone   = db.Column(db.Text)
    email      = db.Column(db.Text)

    projetos   = db.relationship('Projeto', back_populates='empresa', cascade='all, delete-orphan')

class Projeto(db.Model):
    __tablename__ = 'projetos'
    id               = db.Column(db.BigInteger, primary_key=True)
    nome             = db.Column(db.Text)
    empresa_id       = db.Column(db.BigInteger, db.ForeignKey('empresas.id'))
    descricao        = db.Column(db.Text)
    data_criacao     = db.Column(db.DateTime, default=datetime.utcnow)
    data_atualizacao = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status           = db.Column(db.Text)

    empresa     = db.relationship('Empresa', back_populates='projetos')
    responsaveis= db.relationship('ResponsavelProjeto', back_populates='projeto', cascade='all, delete-orphan')
    edificios   = db.relationship('Edificio',     back_populates='projeto', cascade='all, delete-orphan')
    relatorios  = db.relationship('Relatorio',    back_populates='projeto', cascade='all, delete-orphan')
    logs        = db.relationship('LogAlteracao', back_populates='projeto', cascade='all, delete-orphan')

class ResponsavelProjeto(db.Model):
    __tablename__ = 'responsaveis_projeto'
    id          = db.Column(db.BigInteger, primary_key=True)
    projeto_id  = db.Column(db.BigInteger, db.ForeignKey('projetos.id'))
    usuario_id  = db.Column(db.BigInteger, db.ForeignKey('usuarios.id'))

    projeto     = db.relationship('Projeto', back_populates='responsaveis')
    usuario     = db.relationship('Usuario', back_populates='responsaveis')

class Edificio(db.Model):
    __tablename__ = 'edificios'
    id          = db.Column(db.BigInteger, primary_key=True)
    projeto_id  = db.Column(db.BigInteger, db.ForeignKey('projetos.id'))
    nome        = db.Column(db.Text)
    localizacao = db.Column(db.Text)
    tipo        = db.Column(db.Text)
    pavimentos  = db.Column(db.Integer)

    projeto     = db.relationship('Projeto', back_populates='edificios')
    fachadas    = db.relationship('Fachada', back_populates='edificio', cascade='all, delete-orphan')

class Fachada(db.Model):
    __tablename__ = 'fachadas'
    id          = db.Column(db.BigInteger, primary_key=True)
    edificio_id = db.Column(db.BigInteger, db.ForeignKey('edificios.id'))
    nome        = db.Column(db.Text)
    area        = db.Column(DOUBLE_PRECISION)
    descricao   = db.Column(db.Text)

    edificio    = db.relationship('Edificio', back_populates='fachadas')
    imagens     = db.relationship('Imagem',   back_populates='fachada', cascade='all, delete-orphan')

class Imagem(db.Model):
    __tablename__ = 'imagens'
    id             = db.Column(db.BigInteger, primary_key=True)
    fachada_id     = db.Column(db.BigInteger, db.ForeignKey('fachadas.id'))
    caminho_arquivo= db.Column(db.Text)
    nome_arquivo   = db.Column(db.Text)
    data_captura   = db.Column(db.DateTime)
    data_upload    = db.Column(db.DateTime, default=datetime.utcnow)
    metadados      = db.Column(JSON)
    processada     = db.Column(db.Boolean, default=False)

    fachada    = db.relationship('Fachada', back_populates='imagens')
    fissuras   = db.relationship('Fissura', back_populates='imagem', cascade='all, delete-orphan')

class Fissura(db.Model):
    __tablename__ = 'fissuras'
    id            = db.Column(db.BigInteger, primary_key=True)
    imagem_id     = db.Column(db.BigInteger, db.ForeignKey('imagens.id'))
    tipo          = db.Column(db.Text)
    coordenadas   = db.Column(JSON)
    gravidade     = db.Column(db.Text)
    data_deteccao = db.Column(db.DateTime, default=datetime.utcnow)
    confianca     = db.Column(DOUBLE_PRECISION)

    imagem       = db.relationship('Imagem', back_populates='fissuras')

class Relatorio(db.Model):
    __tablename__ = 'relatorios'
    id            = db.Column(db.BigInteger, primary_key=True)
    projeto_id    = db.Column(db.BigInteger, db.ForeignKey('projetos.id'))
    usuario_id    = db.Column(db.BigInteger, db.ForeignKey('usuarios.id'))
    titulo        = db.Column(db.Text)
    caminho_arquivo= db.Column(db.Text)
    data_geracao  = db.Column(db.DateTime, default=datetime.utcnow)
    parametros    = db.Column(JSON)

    projeto       = db.relationship('Projeto', back_populates='relatorios')
    usuario       = db.relationship('Usuario', back_populates='relatorios')

class LogAlteracao(db.Model):
    __tablename__ = 'logs_alteracoes'
    id               = db.Column(db.BigInteger, primary_key=True)
    projeto_id       = db.Column(db.BigInteger, db.ForeignKey('projetos.id'))
    usuario_id       = db.Column(db.BigInteger, db.ForeignKey('usuarios.id'))
    tipo_alteracao   = db.Column(db.Text)
    descricao        = db.Column(db.Text)
    data_alteracao   = db.Column(db.DateTime, default=datetime.utcnow)
    entidade_afetada = db.Column(db.Text)
    entidade_id      = db.Column(db.BigInteger)

    projeto          = db.relationship('Projeto', back_populates='logs')
    usuario          = db.relationship('Usuario', back_populates='logs_alteracoes')
