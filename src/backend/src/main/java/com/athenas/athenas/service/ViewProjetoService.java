package com.athenas.athenas.service;

import java.time.format.DateTimeFormatter;
import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.athenas.athenas.DTO.ViewProjetoResponseDTO;
import com.athenas.athenas.model.Edificio;
import com.athenas.athenas.model.Empresa;
import com.athenas.athenas.model.LogAlteracao;
import com.athenas.athenas.model.Projeto;
import com.athenas.athenas.model.ResponsavelProjeto;
import com.athenas.athenas.model.Usuario;
import com.athenas.athenas.repository.EdificioRepository;
import com.athenas.athenas.repository.LogAlteracaoRepository;
import com.athenas.athenas.repository.ProjetoRepository;
import com.athenas.athenas.repository.ResponsavelProjetoRepository;

@Service
public class ViewProjetoService {

    @Autowired
    private ProjetoRepository projetoRepository;

    @Autowired
    private EdificioRepository edificioRepository;

    @Autowired
    private LogAlteracaoRepository logAlteracaoRepository;

    @Autowired
    private ResponsavelProjetoRepository responsavelProjetoRepository;

    private final DateTimeFormatter formatter = DateTimeFormatter.ofPattern("dd/MM/yyyy HH:mm:ss"); 

    public ViewProjetoResponseDTO ReadProjectData(int idProjeto) {
        ViewProjetoResponseDTO response = new ViewProjetoResponseDTO();

        Projeto projeto = projetoRepository.findById(idProjeto)
            .orElseThrow(() -> new RuntimeException("Projeto not found with id: " + idProjeto));

        response.setNome(projeto.getNome());

        List<ResponsavelProjeto> responsaveisProjeto = responsavelProjetoRepository.findByProjeto(projeto);
        List<String> responsaveisNomes = responsaveisProjeto.stream()
            .map(ResponsavelProjeto::getUsuario)
            .map(Usuario::getNome)
            .toList();
        response.setResponsaveisNomes(responsaveisNomes);

        Empresa empresa = projeto.getEmpresa();
        response.setEmpresaNome(empresa.getNome());

        List<Edificio> edificios = edificioRepository.findByProjeto(projeto)
            .orElseThrow(() -> new RuntimeException("Edificio not found"));
        response.setEdificios(edificios);
        
        response.setDescricao(projeto.getDescricao());

        List<LogAlteracao> logAlteracoes = logAlteracaoRepository.findByProjeto(projeto);
        response.setLogs((List<String>) logAlteracoes.stream()
            .map(log -> (log.getDataAlteracao().format(formatter) + " - " + log.getDescricao()))
            .toList());
        
        return response;
    }
}