package com.athenas.athenas.service;

import java.time.format.DateTimeFormatter;
import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.athenas.athenas.DTO.EdificioDTO;
import com.athenas.athenas.DTO.ViewProjetoResponseDTO;
import com.athenas.athenas.model.Edificio;
import com.athenas.athenas.model.Empresa;
import com.athenas.athenas.model.LogAlteracao;
import com.athenas.athenas.model.Projeto;
import com.athenas.athenas.model.ResponsavelProjeto;
import com.athenas.athenas.model.Usuario;
import com.athenas.athenas.repository.EdificioRepository;
import com.athenas.athenas.repository.EmpresaRepository;
import com.athenas.athenas.repository.LogAlteracaoRepository;
import com.athenas.athenas.repository.ProjetoRepository;
import com.athenas.athenas.repository.ResponsavelProjetoRepository;
import com.athenas.athenas.repository.UsuarioRepository;

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

    @Autowired 
    private EmpresaRepository empresaRepository;

    @Autowired
    private UsuarioRepository usuarioRepository;

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

        List<Edificio> edificios = edificioRepository.findByProjeto(projeto);
        List<EdificioDTO> edificiosDTO = edificios.stream()
            .map(edificio -> {
                EdificioDTO dto = new EdificioDTO();
                dto.setNome(edificio.getNome());
                dto.setLocalizacao(edificio.getLocalizacao());
                dto.setTipo(edificio.getTipo());
                dto.setPavimentos(edificio.getPavimentos());
                return dto;
            })
            .toList();
        response.setEdificios(edificiosDTO); 
        
        response.setDescricao(projeto.getDescricao());

        List<LogAlteracao> logAlteracoes = logAlteracaoRepository.findByProjeto(projeto);
        response.setLogs((List<String>) logAlteracoes.stream()
            .map(log -> (log.getDataAlteracao().format(formatter) + " - " + log.getDescricao()))
            .toList());
        
        return response;
    }

    public ViewProjetoResponseDTO UpdateProjectData(int idProjeto, ViewProjetoResponseDTO viewProjetoResponseDTO){

        Projeto projeto = projetoRepository.findById(idProjeto)
            .orElseThrow(() -> new RuntimeException("Projeto not found with id: " + idProjeto));
            
        if (viewProjetoResponseDTO.getNome() != null && !viewProjetoResponseDTO.getNome().equals(projeto.getNome())) {
            String descricao = "Nome alterado de '" + projeto.getNome() + "' para '" + viewProjetoResponseDTO.getNome() + "'. ";
            createLog(descricao, projeto);
            projeto.setNome(viewProjetoResponseDTO.getNome());
        }
    
        if (viewProjetoResponseDTO.getDescricao() != null && !viewProjetoResponseDTO.getDescricao().equals(projeto.getDescricao())) {
            String descricao = "Descrição alterada de '" + projeto.getDescricao() + "' para '" + viewProjetoResponseDTO.getDescricao() + "'. ";
            createLog(descricao, projeto);
            projeto.setDescricao(viewProjetoResponseDTO.getDescricao());
        }
    
        if (viewProjetoResponseDTO.getEmpresaNome() != null && !viewProjetoResponseDTO.getEmpresaNome().equals(projeto.getEmpresa().getNome())) {
            String descricao = "Empresa alterada de '" + projeto.getEmpresa().getNome() + "' para '" + viewProjetoResponseDTO.getEmpresaNome() + "'. ";
            createLog(descricao, projeto);
            Empresa novaEmpresa = empresaRepository.findByNome(viewProjetoResponseDTO.getEmpresaNome())
                .orElseThrow(() -> new RuntimeException("Empresa não encontrada. Cadastre-a primeiro."));
            projeto.setEmpresa(novaEmpresa);
        }

        // Ver com a cecilia como ta o front pra criar edificio
        // List<Edificio> edificiosAtuais = edificioRepository.findByProjeto(projeto);
        // List<EdificioDTO> novosEdificios = viewProjetoResponseDTO.getEdificios();
        // if (novosEdificios != null && !novosEdificios.isEmpty()) {
        //     for (Edificio edificio : edificiosAtuais) {
        //         edificioRepository.delete(edificio);
        //     }
        //     for (EdificioDTO edificioDTO : novosEdificios) {
        //         Edificio novo = new Edificio();
        //         novo.setNome(edificioDTO.getNome());
        //         novo.setLocalizacao(edificioDTO.getLocalizacao());
        //         novo.setTipo(edificioDTO.getTipo());
        //         novo.setPavimentos(edificioDTO.getPavimentos());
        //         novo.setProjeto(projeto);
        //         edificioRepository.save(novo);
        //     }
            
        // }

        // Lógica de trocar incorreta
        // List<String> novosResponsaveis = viewProjetoResponseDTO.getResponsaveisNomes();
        // if (novosResponsaveis != null && !novosResponsaveis.isEmpty()) {
        //     List<ResponsavelProjeto> antigosResponsaveis = responsavelProjetoRepository.findByProjeto(projeto);
        //     for (ResponsavelProjeto responsavelProjeto : antigosResponsaveis) {
        //         responsavelProjetoRepository.delete(responsavelProjeto);
        //         String descricao = "'" + responsavelProjeto.getUsuario().getNome() + "' removido dos responsáveis pelo projeto.";
        //         createLog(descricao, projeto);
        //     }
        //     for (String nomeUsuario : novosResponsaveis) {
        //         Usuario usuario = usuarioRepository.findByNome(nomeUsuario)
        //             .orElseThrow(() -> new RuntimeException("Usuário não encontrado. Cadastre-o primeiro."));
        //         ResponsavelProjeto responsavelProjeto = new ResponsavelProjeto();
        //         responsavelProjeto.setProjeto(projeto);
        //         responsavelProjeto.setUsuario(usuario);
        //         responsavelProjetoRepository.save(responsavelProjeto);
        //     }
        // }
    
        projetoRepository.save(projeto);

        return ReadProjectData(idProjeto);
    }

    private void createLog(String descricao, Projeto projeto){
        LogAlteracao log = new LogAlteracao();
        log.setProjeto(projeto);
        log.setDescricao(descricao);
        log.setDataAlteracao(java.time.LocalDateTime.now());
        // log.setUsuario(usuario);
        logAlteracaoRepository.save(log);
    }
}